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

from src import analytics, llm_client, prepare_data, tickets

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


def _topic_en(df: pd.DataFrame, topic_ar: str) -> str:
    """Look up the English label for an Arabic topic string from the dataset."""
    if not topic_ar:
        return ""
    if "topic_label_ar" in df.columns and "topic_label_en" in df.columns:
        m = df[df["topic_label_ar"] == topic_ar]
        if not m.empty:
            v = m["topic_label_en"].iloc[0]
            if v:
                return str(v)
    return topic_ar


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
    cats = sorted(df["category"].dropna().unique().tolist())
    return {
        "rows": int(len(df)),
        "date_min": str(df["closed_at"].min().date()),
        "date_max": str(df["closed_at"].max().date()),
        "categories": cats,
        "categories_en": [llm_client.CATEGORY_EN.get(c, c) for c in cats],
        "severities":   analytics.SEVERITY_ORDER,
        "severities_en":[llm_client.SEVERITY_EN.get(s, s) for s in analytics.SEVERITY_ORDER],
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
    return JSONResponse([
        {"label": k,
         "label_en": llm_client.CATEGORY_EN.get(k, k),
         "count": int(v)}
        for k, v in s.items()
    ])


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
    series = [{
        "name": col,
        "name_en": llm_client.CATEGORY_EN.get(col, col),
        "y": [int(v) for v in pivot[col].values],
    } for col in pivot.columns]
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
    rows = df.to_dict(orient="records")
    for r in rows:
        r["topic_label_en"] = _topic_en(fdf, r.get("topic_label", ""))
    return JSONResponse(rows)


@app.get("/api/alerts")
def alerts(
    date_from: Optional[date] = Query(None, alias="from"),
    date_to: Optional[date] = Query(None, alias="to"),
    category: Optional[list[str]] = Query(None),
    severity: Optional[list[str]] = Query(None),
    language: str = Query("ar"),
) -> JSONResponse:
    fdf = _params(date_from, date_to, category, severity)
    out = []
    for a in analytics.detect_weekly_anomalies(fdf, lang=language):
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
    assignee: Optional[list[str]] = Query(None),  # user_id list, "unassigned" allowed
    status: Optional[list[str]] = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=200),
) -> JSONResponse:
    fdf = _params(date_from, date_to, category, severity)
    if topic:
        fdf = fdf[fdf["topic_label"].isin(topic)]
    if body_contains:
        fdf = fdf[fdf["body"].astype(str).str.contains(body_contains, regex=False, na=False)]
    if week_start:
        ws = pd.to_datetime(week_start, errors="coerce")
        if pd.notna(ws):
            fdf = fdf[fdf["week_start"] == ws]

    # Apply ticket-state filters (status / assignee / severity override).
    # Severity needs special handling: the user might filter by the EFFECTIVE
    # severity (override-or-AI), so we compute it before further filtering.
    assignments = {rid: (asn, st, ov)
                   for rid, asn, st, ov in tickets.all_assignments()}

    def effective_severity(rid: int, ai_sev: str) -> str:
        a = assignments.get(rid)
        return (a[2] if a and a[2] else ai_sev) if a else ai_sev

    fdf = fdf.copy()
    if not fdf.empty:
        fdf["effective_severity"] = [
            effective_severity(int(rid), sev)
            for rid, sev in zip(fdf["request_id"].tolist(), fdf["severity"].tolist())
        ]
    else:
        fdf["effective_severity"] = []

    if severity and not fdf.empty:
        fdf = fdf[fdf["effective_severity"].isin(severity)]
    if only_high and not fdf.empty:
        fdf = fdf[fdf["effective_severity"] == "عالية"]
    if assignee and not fdf.empty:
        def match_assignee(rid: int) -> bool:
            a = assignments.get(int(rid))
            cur = a[0] if a else None
            if "unassigned" in assignee and not cur:
                return True
            return cur in assignee
        fdf = fdf[fdf["request_id"].apply(match_assignee)]
    if status and not fdf.empty:
        def match_status(rid: int) -> bool:
            a = assignments.get(int(rid))
            cur = a[1] if a else "open"
            return cur in status
        fdf = fdf[fdf["request_id"].apply(match_status)]

    if not fdf.empty:
        fdf = fdf.sort_values("closed_at", ascending=False)
    total = len(fdf)
    start = (page - 1) * page_size
    page_df = fdf.iloc[start:start + page_size]
    rows = []
    page_ids = [int(r["request_id"]) for _, r in page_df.iterrows()]
    ticket_meta = tickets.get_tickets_summary(page_ids) if page_ids else {}
    for _, r in page_df.iterrows():
        rid = int(r["request_id"])
        tm = ticket_meta.get(rid, {})
        ai_sev = r["severity"]
        override = tm.get("severity_override")
        rows.append({
            "request_id": rid,
            "category": r["category"],
            "category_en": llm_client.CATEGORY_EN.get(r["category"], r["category"]),
            "body": r["body"],
            "topic_label_ar": r.get("topic_label_ar") or r.get("topic_label") or "",
            "topic_label_en": r.get("topic_label_en") or r.get("topic_label") or "",
            "severity_ai":  ai_sev,
            "severity":     override or ai_sev,
            "severity_overridden": bool(override),
            "severity_reason_ar": r.get("severity_reason_ar") or r.get("severity_reason") or "",
            "severity_reason_en": r.get("severity_reason_en") or r.get("severity_reason") or "",
            "recommended_action_ar": r.get("recommended_action_ar") or r.get("recommended_action") or "",
            "recommended_action_en": r.get("recommended_action_en") or r.get("recommended_action") or "",
            "closed_at": pd.Timestamp(r["closed_at"]).strftime("%Y-%m-%d %H:%M"),
            "ticket_status":   tm.get("status", "open"),
            "ticket_assignee": tm.get("assignee_id"),
            "ticket_comments": tm.get("comments", 0),
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

@app.get("/api/dashboard_summary")
def dashboard_summary(
    date_from: Optional[date] = Query(None, alias="from"),
    date_to: Optional[date] = Query(None, alias="to"),
    category: Optional[list[str]] = Query(None),
    severity: Optional[list[str]] = Query(None),
    language: str = Query("ar"),
) -> JSONResponse:
    fdf = _params(date_from, date_to, category, severity)
    return JSONResponse(analytics.dashboard_ai_summary(fdf, lang=language))


@app.get("/api/ticket_ai/{request_id}")
def ticket_ai(request_id: int, language: str = Query("ar")) -> dict:
    df = _load_data()
    rec_rows = df[df["request_id"] == request_id]
    if rec_rows.empty:
        raise HTTPException(status_code=404, detail="record not found")
    rec = rec_rows.iloc[0].to_dict()
    return analytics.ticket_ai_view(rec, df, lang=language)


@app.get("/api/insights")
def insights(
    date_from: Optional[date] = Query(None, alias="from"),
    date_to: Optional[date] = Query(None, alias="to"),
    category: Optional[list[str]] = Query(None),
    severity: Optional[list[str]] = Query(None),
    language: str = Query("ar"),
) -> JSONResponse:
    fdf = _params(date_from, date_to, category, severity)
    if fdf.empty:
        return JSONResponse({"insights": [], "source": "empty"})
    items = analytics.rule_based_insights(fdf, lang=language)
    return JSONResponse({"insights": items, "source": "engine"})


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
        return JSONResponse({"categories": [], "categories_en": [],
                             "severities": [], "severities_en": [], "values": []})
    cats = pivot["category"].tolist()
    return JSONResponse({
        "categories":    cats,
        "categories_en": [llm_client.CATEGORY_EN.get(c, c) for c in cats],
        "severities":    analytics.SEVERITY_ORDER,
        "severities_en": [llm_client.SEVERITY_EN.get(s, s) for s in analytics.SEVERITY_ORDER],
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
    rows = df.to_dict(orient="records")
    for r in rows:
        r["topic_label_en"] = _topic_en(fdf, r.get("topic_label", ""))
    return JSONResponse(rows)


# ---------- period comparison ----------

@app.get("/api/period_comparison")
def period_comparison(
    date_from: Optional[date] = Query(None, alias="from"),
    date_to: Optional[date] = Query(None, alias="to"),
    category: Optional[list[str]] = Query(None),
    severity: Optional[list[str]] = Query(None),
    days: int = Query(14, ge=3, le=90),
) -> JSONResponse:
    fdf = _params(date_from, date_to, category, severity)
    return JSONResponse(analytics.period_comparison(fdf, days=days))


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


@app.get("/api/related_groups")
def related_groups(
    date_from: Optional[date] = Query(None, alias="from"),
    date_to: Optional[date] = Query(None, alias="to"),
    category: Optional[list[str]] = Query(None),
    severity: Optional[list[str]] = Query(None),
    min_size: int = Query(5, ge=2, le=50),
    top_n: int = Query(8, ge=1, le=20),
    language: str = Query("ar"),
    use_llm: bool = Query(False),
) -> JSONResponse:
    """Return semantically-linked complaint groups, each with the synthesised
    'beneficiary intent' and a one-line employee response.

    When use_llm=true and a provider is reachable, the LLM rewrites the intent
    by reading the actual body samples in domain context.
    """
    fdf = _params(date_from, date_to, category, severity)
    return JSONResponse(analytics.find_related_groups(
        fdf, min_size=min_size, top_n=top_n, lang=language, use_llm=use_llm))


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


# ---------- ticket management ----------

@app.get("/api/users")
def list_users() -> dict:
    return {"users": tickets.USERS, "statuses": tickets.STATUSES}


@app.get("/api/ticket/{request_id}")
def get_ticket(request_id: int) -> dict:
    """Return ticket state plus the underlying record (body, severity, etc.)."""
    df = _load_data()
    rec = df[df["request_id"] == request_id]
    if rec.empty:
        raise HTTPException(status_code=404, detail="record not found")
    r = rec.iloc[0]
    record = {
        "request_id": int(r["request_id"]),
        "category": r["category"],
        "category_en": llm_client.CATEGORY_EN.get(r["category"], r["category"]),
        "body": r["body"],
        "topic_label_ar": r.get("topic_label_ar") or r.get("topic_label") or "",
        "topic_label_en": r.get("topic_label_en") or r.get("topic_label") or "",
        "severity": r["severity"],
        "severity_reason_ar": r.get("severity_reason_ar") or r.get("severity_reason") or "",
        "severity_reason_en": r.get("severity_reason_en") or r.get("severity_reason") or "",
        "recommended_action_ar": r.get("recommended_action_ar") or r.get("recommended_action") or "",
        "recommended_action_en": r.get("recommended_action_en") or r.get("recommended_action") or "",
        "closed_at": pd.Timestamp(r["closed_at"]).strftime("%Y-%m-%d %H:%M"),
    }
    return {"record": record, "ticket": tickets.get_ticket(request_id)}


class StatusBody(BaseModel):
    status: str
    by_id: Optional[str] = None


@app.post("/api/ticket/{request_id}/status")
def ticket_status(request_id: int, body: StatusBody) -> dict:
    try:
        return tickets.set_status(request_id, body.status, body.by_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


class AssignBody(BaseModel):
    assignee_id: Optional[str] = None
    by_id: Optional[str] = None


@app.post("/api/ticket/{request_id}/assign")
def ticket_assign(request_id: int, body: AssignBody) -> dict:
    try:
        return tickets.set_assignee(request_id, body.assignee_id, body.by_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


class CommentBody(BaseModel):
    author_id: str
    body: str


@app.post("/api/ticket/{request_id}/comment")
def ticket_comment(request_id: int, body: CommentBody) -> dict:
    try:
        return tickets.add_comment(request_id, body.author_id, body.body)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/ticket_stats")
def ticket_stats() -> dict:
    return tickets.stats()


class SeverityBody(BaseModel):
    severity: Optional[str] = None  # None to clear override
    by_id: Optional[str] = None


@app.post("/api/ticket/{request_id}/severity")
def ticket_severity(request_id: int, body: SeverityBody) -> dict:
    try:
        return tickets.set_severity(request_id, body.severity, body.by_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
