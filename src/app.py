"""FastAPI backend for the customer-experience analytics dashboard.

Serves:
  - GET  /                     → single-page dashboard (templates/index.html)
  - GET  /api/health           → liveness + LLM provider status
  - GET  /api/kpis             → headline metrics
  - GET  /api/categories       → category counts
  - GET  /api/severity         → severity counts
  - GET  /api/weekly           → weekly volume trend
  - GET  /api/weekly_by_cat    → weekly volume per category
  - GET  /api/topics           → top recurring topics
  - GET  /api/alerts           → anomaly alerts (early-warning)
  - GET  /api/records          → paginated records list (drill-down)
  - GET  /api/forecast         → 1-2 week forecast with confidence band
  - GET  /api/recurring_cases  → recurring-text clusters
  - POST /api/qa               → free-form question, answered locally
  - POST /api/upload           → ingest a new .xlsx into the dataset
"""

from __future__ import annotations

import sys
from datetime import date, datetime
from pathlib import Path
from typing import Optional

# Ensure project root is on sys.path so `from src import …` works under uvicorn
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import logging
import re
import shutil
import time

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src import analytics, llm_client, prepare_data

log = logging.getLogger("app")

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed" / "enriched.parquet"
TEMPLATES_DIR = ROOT / "templates"
STATIC_DIR = ROOT / "static"

app = FastAPI(
    title="منصة تحليل مشاركات المستفيدين",
    description="تحليل ذكي لمشاركات المستفيدين — معالجة محلية بالكامل.",
    version="1.2.0",
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------- data loading ----------

_df_cache: Optional[pd.DataFrame] = None
_df_mtime: float = 0.0


def _load_data() -> pd.DataFrame:
    """Return the enriched dataframe. Auto-reloads when the parquet changes."""
    global _df_cache, _df_mtime
    if not PROCESSED.exists():
        raise HTTPException(
            status_code=503,
            detail="لم يتم تحضير البيانات بعد. ارفع ملف Excel أو شغّل خطوة التحضير.",
        )
    mtime = PROCESSED.stat().st_mtime
    if _df_cache is None or mtime != _df_mtime:
        df = pd.read_parquet(PROCESSED)
        df["closed_at"] = pd.to_datetime(df["closed_at"])
        df["week_start"] = pd.to_datetime(df["week_start"])
        _df_cache = df
        _df_mtime = mtime
    return _df_cache


def _invalidate_cache() -> None:
    global _df_cache, _df_mtime
    _df_cache = None
    _df_mtime = 0.0


_SAFE_NAME = re.compile(r"[^A-Za-z0-9._؀-ۿ\- ]+")


def _safe_filename(name: str) -> str:
    """Normalize a user-supplied filename to keep RAW_DIR tidy."""
    base = Path(name).name
    base = _SAFE_NAME.sub("_", base).strip()
    if not base.lower().endswith(".xlsx"):
        base += ".xlsx"
    return base or f"upload_{int(time.time())}.xlsx"


def _filter_df(
    df: pd.DataFrame,
    *,
    date_from: Optional[date] = None,
    date_to: Optional[date] = None,
    categories: Optional[list[str]] = None,
    severities: Optional[list[str]] = None,
) -> pd.DataFrame:
    out = df
    if date_from is not None:
        out = out[out["closed_at"].dt.date >= date_from]
    if date_to is not None:
        out = out[out["closed_at"].dt.date <= date_to]
    if categories:
        out = out[out["category"].isin(categories)]
    if severities:
        out = out[out["severity"].isin(severities)]
    return out


def _params(
    date_from: Optional[date],
    date_to: Optional[date],
    category: Optional[list[str]],
    severity: Optional[list[str]],
) -> pd.DataFrame:
    return _filter_df(
        _load_data(),
        date_from=date_from,
        date_to=date_to,
        categories=category,
        severities=severity,
    )


# ---------- root + health ----------

@app.get("/", response_class=HTMLResponse)
def root() -> HTMLResponse:
    index = TEMPLATES_DIR / "index.html"
    if not index.exists():
        raise HTTPException(status_code=500, detail="index.html missing")
    return HTMLResponse(index.read_text(encoding="utf-8"))


@app.get("/api/health")
def health() -> dict:
    try:
        df = _load_data()
        ready = True
        rows = len(df)
    except HTTPException:
        ready = False
        rows = 0
    return {
        "ready": ready,
        "rows": rows,
        **llm_client.runtime_status(),
        "now": datetime.utcnow().isoformat() + "Z",
    }


@app.get("/api/meta")
def meta() -> dict:
    """Metadata used by the frontend to populate filter widgets."""
    df = _load_data()
    return {
        "rows": int(len(df)),
        "date_min": str(df["closed_at"].min().date()),
        "date_max": str(df["closed_at"].max().date()),
        "categories": sorted(df["category"].dropna().unique().tolist()),
        "severities": analytics.SEVERITY_ORDER,
    }


# ---------- analytics endpoints ----------

@app.get("/api/kpis")
def kpis(
    date_from: Optional[date] = Query(None, alias="from"),
    date_to: Optional[date] = Query(None, alias="to"),
    category: Optional[list[str]] = Query(None),
    severity: Optional[list[str]] = Query(None),
) -> dict:
    fdf = _params(date_from, date_to, category, severity)
    k = analytics.compute_kpis(fdf)
    return {
        "total": int(k.total),
        "pct_complaints": round(k.pct_complaints, 1),
        "pct_high_severity": round(k.pct_high_severity, 1),
        "active_categories": int((k.by_category > 0).sum()),
        # Weekly delta — last week vs prior 4-week average
        "weekly_delta_pct": _weekly_delta(k.weekly_volume),
    }


def _weekly_delta(weekly: pd.Series) -> Optional[float]:
    if len(weekly) < 5:
        return None
    last = float(weekly.iloc[-1])
    base = float(weekly.iloc[-5:-1].mean())
    if base == 0:
        return None
    return round((last - base) / base * 100.0, 1)


@app.get("/api/categories")
def categories(
    date_from: Optional[date] = Query(None, alias="from"),
    date_to: Optional[date] = Query(None, alias="to"),
    category: Optional[list[str]] = Query(None),
    severity: Optional[list[str]] = Query(None),
) -> JSONResponse:
    fdf = _params(date_from, date_to, category, severity)
    s = fdf["category"].value_counts()
    return JSONResponse([{"label": k, "count": int(v)} for k, v in s.items()])


@app.get("/api/severity")
def severity_split(
    date_from: Optional[date] = Query(None, alias="from"),
    date_to: Optional[date] = Query(None, alias="to"),
    category: Optional[list[str]] = Query(None),
    severity: Optional[list[str]] = Query(None),
) -> JSONResponse:
    fdf = _params(date_from, date_to, category, severity)
    s = fdf["severity"].value_counts().reindex(analytics.SEVERITY_ORDER, fill_value=0)
    return JSONResponse([{"label": k, "count": int(v)} for k, v in s.items()])


@app.get("/api/weekly")
def weekly(
    date_from: Optional[date] = Query(None, alias="from"),
    date_to: Optional[date] = Query(None, alias="to"),
    category: Optional[list[str]] = Query(None),
    severity: Optional[list[str]] = Query(None),
) -> JSONResponse:
    fdf = _params(date_from, date_to, category, severity)
    if fdf.empty:
        return JSONResponse({"x": [], "y": []})
    g = fdf.groupby(fdf["week_start"]).size().sort_index()
    return JSONResponse({
        "x": [d.strftime("%Y-%m-%d") for d in g.index],
        "y": [int(v) for v in g.values],
    })


@app.get("/api/weekly_by_cat")
def weekly_by_cat(
    date_from: Optional[date] = Query(None, alias="from"),
    date_to: Optional[date] = Query(None, alias="to"),
    category: Optional[list[str]] = Query(None),
    severity: Optional[list[str]] = Query(None),
) -> JSONResponse:
    fdf = _params(date_from, date_to, category, severity)
    if fdf.empty:
        return JSONResponse({"x": [], "series": []})
    pivot = (
        fdf.groupby([fdf["week_start"], "category"]).size().unstack(fill_value=0).sort_index()
    )
    x = [d.strftime("%Y-%m-%d") for d in pivot.index]
    series = [{"name": col, "y": [int(v) for v in pivot[col].values]} for col in pivot.columns]
    return JSONResponse({"x": x, "series": series})


@app.get("/api/topics")
def topics(
    date_from: Optional[date] = Query(None, alias="from"),
    date_to: Optional[date] = Query(None, alias="to"),
    category: Optional[list[str]] = Query(None),
    severity: Optional[list[str]] = Query(None),
    top_n: int = Query(10, ge=1, le=30),
) -> JSONResponse:
    fdf = _params(date_from, date_to, category, severity)
    df = analytics.top_recurring_topics(fdf, top_n=top_n)
    return JSONResponse(df.to_dict(orient="records"))


@app.get("/api/alerts")
def alerts(
    date_from: Optional[date] = Query(None, alias="from"),
    date_to: Optional[date] = Query(None, alias="to"),
    category: Optional[list[str]] = Query(None),
    severity: Optional[list[str]] = Query(None),
) -> JSONResponse:
    fdf = _params(date_from, date_to, category, severity)
    out = []
    for a in analytics.detect_weekly_anomalies(fdf):
        out.append({
            "week_start": a.week_start.strftime("%Y-%m-%d"),
            "dimension": a.dimension,
            "value": a.value,
            "count": a.count,
            "baseline_mean": round(a.baseline_mean, 1),
            "baseline_std": round(a.baseline_std, 1),
            "z_score": round(a.z_score, 2),
            "severity": a.severity,
            "suggested_action": a.suggested_action,
        })
    return JSONResponse(out)


@app.get("/api/records")
def records(
    date_from: Optional[date] = Query(None, alias="from"),
    date_to: Optional[date] = Query(None, alias="to"),
    category: Optional[list[str]] = Query(None),
    severity: Optional[list[str]] = Query(None),
    topic: Optional[list[str]] = Query(None),
    body_contains: Optional[str] = Query(None),
    week_start: Optional[str] = Query(None),
    only_high: bool = False,
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=200),
) -> JSONResponse:
    fdf = _params(date_from, date_to, category, severity)
    if only_high:
        fdf = fdf[fdf["severity"] == "عالية"]
    if topic:
        fdf = fdf[fdf["topic_label"].isin(topic)]
    if body_contains:
        fdf = fdf[fdf["body"].astype(str).str.contains(body_contains, regex=False, na=False)]
    if week_start:
        ws = pd.to_datetime(week_start, errors="coerce")
        if pd.notna(ws):
            fdf = fdf[fdf["week_start"] == ws]
    fdf = fdf.sort_values("closed_at", ascending=False)
    total = len(fdf)
    start = (page - 1) * page_size
    page_df = fdf.iloc[start:start + page_size]
    rows = []
    for _, r in page_df.iterrows():
        rows.append({
            "request_id": int(r["request_id"]),
            "category": r["category"],
            "body": r["body"],
            "topic_label": r["topic_label"],
            "severity": r["severity"],
            "severity_reason": r.get("severity_reason") or "",
            "recommended_action": r["recommended_action"],
            "closed_at": pd.Timestamp(r["closed_at"]).strftime("%Y-%m-%d %H:%M"),
        })
    return JSONResponse({
        "total": total,
        "page": page,
        "page_size": page_size,
        "rows": rows,
    })


# ---------- free-form QA ----------

class QARequest(BaseModel):
    question: str
    date_from: Optional[date] = None
    date_to: Optional[date] = None
    category: Optional[list[str]] = None
    severity: Optional[list[str]] = None
    language: Optional[str] = "ar"


@app.post("/api/qa")
def qa(req: QARequest) -> dict:
    fdf = _params(req.date_from, req.date_to, req.category, req.severity)
    ctx = analytics.summarize_for_qa(fdf)
    res = llm_client.answer_question(req.question, ctx, language=(req.language or "ar"))
    return {"answer": res["answer"], "source": res["source"], "context": ctx}


# ---------- corpus-level insights ----------

@app.get("/api/insights")
def insights(
    date_from: Optional[date] = Query(None, alias="from"),
    date_to: Optional[date] = Query(None, alias="to"),
    category: Optional[list[str]] = Query(None),
    severity: Optional[list[str]] = Query(None),
) -> JSONResponse:
    """Return 3–4 actionable insights derived from the filtered slice.

    Uses the local LLM if available, otherwise deterministic rule-based
    insights generated from the same signals.
    """
    fdf = _params(date_from, date_to, category, severity)
    if fdf.empty:
        return JSONResponse({"insights": [], "source": "empty"})

    signals = analytics.build_signals_text(fdf)

    # Prefer LLM if reachable; fall back to rule-based insights synthesised
    # from the same signal pack.
    res = llm_client.generate_insights(signals)
    if res["insights"]:
        return JSONResponse({"insights": res["insights"], "source": res["source"], "signals": signals})
    rule_items = analytics.rule_based_insights(fdf)
    return JSONResponse({"insights": rule_items, "source": "rule", "signals": signals})


@app.get("/api/severity_weekly")
def severity_weekly(
    date_from: Optional[date] = Query(None, alias="from"),
    date_to: Optional[date] = Query(None, alias="to"),
    category: Optional[list[str]] = Query(None),
    severity: Optional[list[str]] = Query(None),
) -> JSONResponse:
    fdf = _params(date_from, date_to, category, severity)
    pivot = analytics.severity_by_week(fdf)
    if pivot.empty:
        return JSONResponse({"x": [], "series": []})
    x = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in pivot["week_start"]]
    series = [
        {"name": s, "y": [int(v) for v in pivot[s].values]}
        for s in analytics.SEVERITY_ORDER
    ]
    return JSONResponse({"x": x, "series": series})


@app.get("/api/category_matrix")
def category_matrix(
    date_from: Optional[date] = Query(None, alias="from"),
    date_to: Optional[date] = Query(None, alias="to"),
    category: Optional[list[str]] = Query(None),
    severity: Optional[list[str]] = Query(None),
) -> JSONResponse:
    fdf = _params(date_from, date_to, category, severity)
    pivot = analytics.category_severity_matrix(fdf)
    if pivot.empty:
        return JSONResponse({"categories": [], "severities": [], "values": []})
    return JSONResponse({
        "categories": pivot["category"].tolist(),
        "severities": analytics.SEVERITY_ORDER,
        "values": [[int(pivot[s].iloc[i]) for s in analytics.SEVERITY_ORDER]
                   for i in range(len(pivot))],
    })


@app.get("/api/topic_momentum")
def topic_momentum_endpoint(
    date_from: Optional[date] = Query(None, alias="from"),
    date_to: Optional[date] = Query(None, alias="to"),
    category: Optional[list[str]] = Query(None),
    severity: Optional[list[str]] = Query(None),
) -> JSONResponse:
    fdf = _params(date_from, date_to, category, severity)
    df = analytics.topic_momentum(fdf, lookback_weeks=4, top_n=8)
    return JSONResponse(df.to_dict(orient="records"))


# ---------- forecasting + repeat-case intelligence ----------

@app.get("/api/forecast")
def forecast(
    date_from: Optional[date] = Query(None, alias="from"),
    date_to: Optional[date] = Query(None, alias="to"),
    category: Optional[list[str]] = Query(None),
    severity: Optional[list[str]] = Query(None),
    horizon: int = Query(2, ge=1, le=4),
) -> JSONResponse:
    fdf = _params(date_from, date_to, category, severity)
    return JSONResponse(analytics.forecast_weekly(fdf, horizon=horizon, by_category=True))


@app.get("/api/recurring_cases")
def recurring_cases(
    date_from: Optional[date] = Query(None, alias="from"),
    date_to: Optional[date] = Query(None, alias="to"),
    category: Optional[list[str]] = Query(None),
    severity: Optional[list[str]] = Query(None),
    min_repeats: int = Query(3, ge=2, le=20),
    lookback_days: int = Query(60, ge=7, le=365),
) -> JSONResponse:
    fdf = _params(date_from, date_to, category, severity)
    df = analytics.find_recurring_cases(fdf, min_repeats=min_repeats, lookback_days=lookback_days)
    return JSONResponse(df.head(20).to_dict(orient="records"))


# ---------- ingestion: upload + refresh ----------

ALLOWED_UPLOAD_TYPES = {
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.ms-excel",
    "application/octet-stream",   # Safari sometimes sends this
}


@app.post("/api/upload")
async def upload_excel(file: UploadFile = File(...)) -> dict:
    """Accept a new .xlsx export, append it to the raw store, and re-enrich."""
    if not file.filename or not file.filename.lower().endswith(".xlsx"):
        raise HTTPException(status_code=400, detail="يرجى رفع ملف بصيغة .xlsx")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    target = RAW_DIR / _safe_filename(file.filename)

    # Avoid silently overwriting an existing file with the same name
    if target.exists():
        stem = target.stem
        target = RAW_DIR / f"{stem}_{int(time.time())}.xlsx"

    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="الملف فارغ")
    if len(contents) > 50 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="حجم الملف أكبر من 50 ميغابايت")

    target.write_bytes(contents)
    log.info("saved upload to %s (%d bytes)", target, len(contents))

    try:
        # Validate the new file alone first so we can give a precise error
        # if its schema is wrong, BEFORE rebuilding the full cache.
        prepare_data._load_one(target)
    except Exception as exc:
        target.unlink(missing_ok=True)
        raise HTTPException(
            status_code=400,
            detail=f"تعذّر قراءة الملف: {exc}. يجب أن يحوي الأعمدة "
                   "(الرقم، الطلب، العنوان، نص الطلب، تاريخ الانتهاء).",
        )

    try:
        summary = prepare_data.run()
    except Exception as exc:
        log.exception("prepare_data failed after upload")
        raise HTTPException(status_code=500, detail=f"فشل تحضير البيانات: {exc}")

    _invalidate_cache()
    return {
        "status": "ok",
        "saved_as": target.name,
        "summary": summary,
    }


@app.post("/api/refresh")
def refresh() -> dict:
    """Re-run enrichment over the current raw files (no upload)."""
    try:
        summary = prepare_data.run()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    _invalidate_cache()
    return {"status": "ok", "summary": summary}


@app.get("/api/files")
def list_files() -> dict:
    """List the Excel files currently feeding the dashboard."""
    if not RAW_DIR.exists():
        return {"files": []}
    items = []
    for p in sorted(RAW_DIR.glob("*.xlsx")):
        if p.name.startswith("~$"):
            continue
        st = p.stat()
        items.append({
            "name": p.name,
            "size_kb": round(st.st_size / 1024, 1),
            "uploaded_at": datetime.utcfromtimestamp(st.st_mtime).isoformat() + "Z",
        })
    return {"files": items}
