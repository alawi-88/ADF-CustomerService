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
    title="منصّة تحليل مشاركات المستفيدين — صندوق التنمية الزراعية",
    description="تحليل ذكي لمشاركات المستفيدين عبر القنوات الرقمية — معالجة محلية بالكامل.",
    version="1.7.0",
)


# --- security middleware (v1.2.1) -----------------------------------------
# Restrict CORS by default to same-origin. Operators can override via env.
import os as _os
from fastapi.middleware.cors import CORSMiddleware
_allowed_origins_raw = _os.environ.get("ADF_CS_CORS_ORIGINS", "").strip()
if _allowed_origins_raw:
    _allowed_origins = [o.strip() for o in _allowed_origins_raw.split(",") if o.strip()]
else:
    # Default: only same-origin requests; the dashboard ships its own HTML.
    _allowed_origins = []
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# --- generic exception handler --------------------------------------------
# Without this, FastAPI exposes the full Python traceback for unexpected
# errors. We log server-side and return a stable, opaque message.
from fastapi.requests import Request as _Request
from fastapi.responses import JSONResponse as _JSONResponse


@app.exception_handler(Exception)
async def _adf_unhandled_exception_handler(request: _Request, exc: Exception):
    log.exception("unhandled error on %s %s", request.method, request.url.path)
    return _JSONResponse(
        status_code=500,
        content={
            "detail": "حدث خطأ غير متوقع. تم تسجيل المشكلة — يرجى المحاولة مجدداً.",
            "error_id": str(int(__import__("time").time() * 1000)),
        },
    )
# --------------------------------------------------------------------------


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
    n_imputed = int(df["_date_missing"].sum()) if "_date_missing" in df.columns else 0
    return {
        "rows": int(len(df)),
        "date_min": str(df["closed_at"].min().date()),
        "date_max": str(df["closed_at"].max().date()),
        "categories": cats,
        "categories_en": [llm_client.CATEGORY_EN.get(c, c) for c in cats],
        "severities":   analytics.SEVERITY_ORDER,
        "severities_en":[llm_client.SEVERITY_EN.get(s, s) for s in analytics.SEVERITY_ORDER],
        "data_quality": {
            "imputed_dates": n_imputed,
            "ingested": int(len(df)),
        },
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
        # New v1.2.0: % of tickets where the body is too short to classify
        "pct_insufficient_context": round(getattr(k, "pct_insufficient_context", 0.0), 1),
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
        log.exception("prepare_data failed in upload handler")
        raise HTTPException(
            status_code=500,
            detail="فشل تحضير البيانات. تم تسجيل الخطأ — يرجى مراجعة سجلات الخادم."
        )

    _invalidate_cache()

    # Auto-snapshot on every successful upload — so management has an
    # audit-grade record of "what the dashboard said when this file landed".
    auto_snapshot_id = None
    try:
        from src import recommendations as _recs
        fdf = _filtered_df_from_filters({})
        k = analytics.compute_kpis(fdf)
        kpis_payload = {
            "total": int(k.total),
            "pct_complaints": round(k.pct_complaints, 1),
            "pct_high_severity": round(k.pct_high_severity, 1),
            "active_categories": int((k.by_category > 0).sum()),
            "weekly_delta_pct": _weekly_delta(k.weekly_volume),
            "pct_insufficient_context": round(getattr(k, "pct_insufficient_context", 0.0), 1),
        }
        insights_items = analytics.rule_based_insights(fdf, lang="ar") if not fdf.empty else []
        forecast_payload = analytics.forecast_weekly(fdf, horizon=2, by_category=True) if len(fdf) else None
        auto_snapshot_id = _recs.create_snapshot({
            "trigger":        "upload",
            "source_file":    target.name,
            "row_count":      int(len(fdf)),
            "filters":        {},
            "provider":       _snapshot_active_provider(),
            "model":          _snapshot_active_model(),
            "prompt_version": getattr(llm_client, "PROMPT_VERSION", "v1"),
            "language":       "ar",
            "insights":       {"insights": insights_items, "source": "engine" if insights_items else "empty"},
            "kpis":           kpis_payload,
            "forecast":       forecast_payload,
            "items":          _items_for_df(fdf),
        })
        log.info("auto-snapshot %s created on upload of %s", auto_snapshot_id, target.name)
    except Exception:
        # An upload should not fail because the audit-log step crashed.
        log.exception("auto-snapshot on upload failed (non-fatal)")

    return {
        "status": "ok",
        "saved_as": target.name,
        "summary": summary,
        "snapshot_id": auto_snapshot_id,
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




@app.get("/api/subcategories")
def subcategories(
    date_from: Optional[date] = Query(None, alias="from"),
    date_to: Optional[date] = Query(None, alias="to"),
    category: Optional[list[str]] = Query(None),
    severity: Optional[list[str]] = Query(None),
) -> dict:
    """Aggregate counts by (category, subcategory).

    Until the LLM is connected most rows carry `subcategory='unspecified'`
    — the response includes a `coverage` field telling the UI what share
    of rows have a meaningful subcategory.
    """
    fdf = _params(date_from, date_to, category, severity)
    if fdf.empty or "subcategory" not in fdf.columns:
        return {"items": [], "coverage": 0.0, "total": 0}

    sub_col = fdf["subcategory"].fillna("unspecified")
    counts = (
        fdf.assign(subcategory=sub_col)
        .groupby(["category", "subcategory"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    items = [
        {
            "category": str(r["category"]),
            "subcategory": str(r["subcategory"]),
            "count": int(r["count"]),
        }
        for _, r in counts.iterrows()
    ]
    coverage = float((sub_col != "unspecified").sum()) / len(fdf) * 100.0
    return {"items": items, "coverage": round(coverage, 1), "total": int(len(fdf))}


# >>> ADF v1.1.0 SNAPSHOT ENDPOINTS — DO NOT EDIT INSIDE THIS BLOCK >>>
@app.get("/favicon.ico", include_in_schema=False)
def _favicon():
    # Silence the noisy console 404 — UI does not currently ship a favicon.
    from fastapi.responses import Response
    return Response(status_code=204)



from src import recommendations as recs


class SnapshotCreateBody(BaseModel):
    trigger: str = "manual"
    source_file: Optional[str] = None
    filters: dict = {}
    language: str = "ar"
    created_by_id: Optional[str] = None


class SnapshotLockBody(BaseModel):
    by_id: str
    note: str = ""


def _snapshot_active_provider() -> str:
    if llm_client.groq_available():
        return "groq"
    if llm_client.ollama_available():
        return "ollama"
    return "rule"


def _snapshot_active_model() -> Optional[str]:
    if llm_client.groq_available():
        models = getattr(llm_client, "GROQ_MODELS", None) or []
        return models[0] if models else None
    if llm_client.ollama_available():
        return getattr(llm_client, "OLLAMA_MODEL", None)
    return None


def _filtered_df_from_filters(filters: dict) -> "pd.DataFrame":
    """Translate the JSON filter shape into the internal _filter_df call."""
    f = filters or {}
    df = _load_data()

    def _to_date(v):
        if not v:
            return None
        if isinstance(v, str):
            try:
                return date.fromisoformat(v[:10])
            except ValueError:
                return None
        return v

    return _filter_df(
        df,
        date_from=_to_date(f.get("from")),
        date_to=_to_date(f.get("to")),
        categories=f.get("category") or None,
        severities=f.get("severity") or None,
    )


def _items_for_df(df) -> list[dict]:
    out = []
    for r in df.to_dict("records"):
        body = r.get("body") or r.get("subject") or ""
        out.append({
            "kind": "ticket",
            "request_id": int(r["request_id"]) if r.get("request_id") is not None else None,
            "category": r.get("category"),
            "subcategory": r.get("subcategory"),
            "severity_ai": r.get("severity"),
            "severity_reason": r.get("severity_reason") or r.get("severity_reason_ar"),
            "topic_label": r.get("topic_label") or r.get("topic_label_ar"),
            "action": r.get("recommended_action") or r.get("recommended_action_ar"),
            "evidence": str(body)[:200],
        })
    return out


@app.post("/api/snapshots")
def api_create_snapshot(body: SnapshotCreateBody):
    fdf = _filtered_df_from_filters(body.filters or {})
    k = analytics.compute_kpis(fdf)
    kpis_payload = {
        "total": int(k.total),
        "pct_complaints": round(k.pct_complaints, 1),
        "pct_high_severity": round(k.pct_high_severity, 1),
        "active_categories": int((k.by_category > 0).sum()),
        "weekly_delta_pct": _weekly_delta(k.weekly_volume),
            "pct_insufficient_context": round(getattr(k, "pct_insufficient_context", 0.0), 1),
        }
    insights_items = analytics.rule_based_insights(fdf, lang=body.language) if not fdf.empty else []
    insights_payload = {"insights": insights_items, "source": "engine" if insights_items else "empty"}
    forecast_payload = analytics.forecast_weekly(fdf, horizon=2, by_category=True) if len(fdf) else None

    snapshot_id = recs.create_snapshot({
        "trigger":        body.trigger,
        "source_file":    body.source_file,
        "row_count":      int(len(fdf)),
        "date_from":      (body.filters or {}).get("from"),
        "date_to":        (body.filters or {}).get("to"),
        "filters":        body.filters,
        "provider":       _snapshot_active_provider(),
        "model":          _snapshot_active_model(),
        "prompt_version": getattr(llm_client, "PROMPT_VERSION", "v1"),
        "language":       body.language,
        "insights":       insights_payload,
        "kpis":           kpis_payload,
        "forecast":       forecast_payload,
        "items":          _items_for_df(fdf),
        "created_by_id":  body.created_by_id,
    })
    return {"snapshot_id": snapshot_id}


@app.get("/api/snapshots")
def api_list_snapshots(limit: int = 50):
    return {"snapshots": recs.list_snapshots(limit=limit)}


@app.get("/api/snapshots/{snapshot_id}")
def api_get_snapshot(snapshot_id: int):
    snap = recs.get_snapshot(snapshot_id)
    if not snap:
        raise HTTPException(status_code=404, detail="snapshot_not_found")
    return snap


@app.post("/api/snapshots/{snapshot_id}/lock")
def api_lock_snapshot(snapshot_id: int, body: SnapshotLockBody):
    recs.lock_snapshot(snapshot_id, by_id=body.by_id, note=body.note)
    return {"ok": True}


@app.get("/api/snapshots/{a_id}/diff/{b_id}")
def api_diff_snapshots(a_id: int, b_id: int):
    try:
        return recs.diff_snapshots(a_id, b_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
# <<< ADF v1.1.0 SNAPSHOT ENDPOINTS <<<


# =============================================================================
# >>> ADF v1.3.0 — Excel export endpoints                                   >>>
# =============================================================================
# Four formatted-with-charts exports, one per user-facing report surface.
# Each endpoint reuses the same `_params(...)` filter helper so the exported
# slice exactly matches what the user sees on screen.
#
# Files come back as application/vnd.openxmlformats-officedocument.spreadsheetml.sheet
# with a Content-Disposition that suggests an ADF-branded filename.
# =============================================================================

from fastapi.responses import StreamingResponse  # noqa: E402
from src import excel_export  # noqa: E402

_XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def _xlsx_response(data: bytes, *, report: str, lang: str) -> StreamingResponse:
    fname = excel_export.filename_for(report, lang=lang)
    return StreamingResponse(
        iter([data]),
        media_type=_XLSX_MIME,
        headers={
            "Content-Disposition": f'attachment; filename="{fname}"',
            "Content-Length": str(len(data)),
            "Cache-Control": "no-store",
        },
    )


def _filter_summary(date_from, date_to, category, severity, lang: str) -> dict:
    """Human-readable summary of the active filters for the cover sheet."""
    if lang.startswith("ar"):
        s = {
            "النطاق الزمني": (
                f"{date_from or '—'} → {date_to or '—'}"
                if (date_from or date_to) else "كامل البيانات"
            ),
            "الفئات": "، ".join(category) if category else "كل الفئات",
            "مستويات الخطورة": "، ".join(severity) if severity else "كل المستويات",
        }
    else:
        s = {
            "Date range": (
                f"{date_from or '—'} → {date_to or '—'}"
                if (date_from or date_to) else "All data"
            ),
            "Categories": ", ".join(category) if category else "All categories",
            "Severity levels": ", ".join(severity) if severity else "All levels",
        }
    return s


# --- helpers that mirror the JSON-route inline computations ---------------

def _export_weekly(fdf: pd.DataFrame) -> list[dict]:
    if fdf.empty:
        return []
    g = fdf.groupby(fdf["week_start"]).size().sort_index()
    return [{"week": d.strftime("%Y-%m-%d"), "count": int(v)} for d, v in g.items()]


def _export_categories(fdf: pd.DataFrame) -> list[dict]:
    if fdf.empty:
        return []
    s = fdf["category"].value_counts()
    return [{"name": k, "count": int(v)} for k, v in s.items()]


def _export_severity(fdf: pd.DataFrame) -> list[dict]:
    if fdf.empty:
        return []
    s = fdf["severity"].value_counts().reindex(analytics.SEVERITY_ORDER, fill_value=0)
    return [{"severity": k, "count": int(v)} for k, v in s.items()]


def _export_weekly_by_cat(fdf: pd.DataFrame) -> list[dict]:
    if fdf.empty:
        return []
    pivot = (
        fdf.groupby([fdf["week_start"], "category"]).size().unstack(fill_value=0).sort_index()
    )
    out = []
    for d, row in pivot.iterrows():
        for col, v in row.items():
            if int(v):
                out.append({"week": d.strftime("%Y-%m-%d"), "category": col, "count": int(v)})
    return out


def _export_topics(fdf: pd.DataFrame, top_n: int = 10) -> list[dict]:
    if fdf.empty:
        return []
    try:
        tdf = analytics.top_recurring_topics(fdf, top_n=top_n)
    except Exception:
        return []
    rows = []
    for _, r in tdf.iterrows():
        rows.append({
            "name":       r.get("topic_label", ""),
            "count":      int(r.get("count", 0) or 0),
            "high_count": int(r.get("high_count", 0) or 0),
        })
    return rows


def _export_severity_weekly(fdf: pd.DataFrame) -> list[dict]:
    if fdf.empty:
        return []
    try:
        sdf = analytics.severity_by_week(fdf)
    except Exception:
        return []
    rows = []
    for _, r in sdf.iterrows():
        rows.append({
            "week": (r.get("week_start").strftime("%Y-%m-%d")
                     if hasattr(r.get("week_start"), "strftime")
                     else str(r.get("week_start", ""))),
            "high": int(r.get("high", 0) or 0),
            "med":  int(r.get("med", 0) or 0),
            "low":  int(r.get("low", 0) or 0),
        })
    return rows


def _export_momentum(fdf: pd.DataFrame) -> list[dict]:
    if fdf.empty:
        return []
    try:
        mdf = analytics.topic_momentum(fdf)
    except Exception:
        return []
    rows = []
    for _, r in mdf.iterrows():
        rows.append({
            "topic":  r.get("topic_label", "") or r.get("topic", ""),
            "recent": int(r.get("recent", 0) or 0),
            "prior":  int(r.get("prior", 0) or 0),
            "delta":  int(r.get("delta", 0) or 0),
        })
    return rows


def _export_alerts(fdf: pd.DataFrame) -> list[dict]:
    if fdf.empty:
        return []
    try:
        alerts_list = analytics.detect_weekly_anomalies(fdf)
    except Exception:
        return []
    out: list[dict] = []
    for a in alerts_list or []:
        # AnomalyAlert is a dataclass — translate to the dict shape the
        # Excel export module expects (title / kind / metric / evidence).
        try:
            week = a.week_start.strftime("%Y-%m-%d") if hasattr(a.week_start, "strftime") else str(a.week_start)
            out.append({
                "title":   f"{a.value} ({a.dimension})",
                "kind":    a.severity,
                "metric":  f"عدد={a.count} · z={a.z_score:.2f} · أسبوع {week}",
                "evidence": a.suggested_action or "",
            })
        except AttributeError:
            # Defensive: if a future change makes it dict-like, fall through.
            if isinstance(a, dict):
                out.append({
                    "title":   a.get("title") or a.get("value") or "",
                    "kind":    a.get("kind")  or a.get("severity") or "",
                    "metric":  a.get("metric") or "",
                    "evidence": a.get("evidence") or a.get("suggested_action") or "",
                })
    return out


def _kpis_for_export(fdf: pd.DataFrame) -> dict:
    try:
        k = analytics.compute_kpis(fdf)
    except Exception:
        return {"total": 0, "complaints_pct": 0, "high_severity": 0,
                "active_categories": 0, "insufficient_pct": 0}
    return {
        "total":              int(k.total),
        "complaints_pct":     round(k.pct_complaints, 1),
        "high_severity":      int((fdf["severity"] == "high").sum()) if not fdf.empty else 0,
        "active_categories":  int((k.by_category > 0).sum()) if hasattr(k, "by_category") else 0,
        "insufficient_pct":   round(getattr(k, "pct_insufficient_context", 0.0), 1),
    }


@app.get("/api/export/overview")
def export_overview(
    date_from: Optional[date] = Query(None, alias="from"),
    date_to:   Optional[date] = Query(None, alias="to"),
    category:  Optional[list[str]] = Query(None),
    severity:  Optional[list[str]] = Query(None),
    lang: str = Query("ar"),
):
    df = _params(date_from, date_to, category, severity)

    data = excel_export.build_overview_workbook(
        df=df,
        kpis=_kpis_for_export(df),
        weekly=_export_weekly(df),
        categories=_export_categories(df),
        severity=_export_severity(df),
        alerts=_export_alerts(df),
        filter_summary=_filter_summary(date_from, date_to, category, severity, lang),
        lang=lang,
    )
    return _xlsx_response(data, report="overview", lang=lang)


@app.get("/api/export/patterns")
def export_patterns(
    date_from: Optional[date] = Query(None, alias="from"),
    date_to:   Optional[date] = Query(None, alias="to"),
    category:  Optional[list[str]] = Query(None),
    severity:  Optional[list[str]] = Query(None),
    lang: str = Query("ar"),
):
    df = _params(date_from, date_to, category, severity)

    data = excel_export.build_patterns_workbook(
        weekly=_export_weekly(df),
        categories=_export_categories(df),
        severity=_export_severity(df),
        severity_weekly=_export_severity_weekly(df),
        momentum=_export_momentum(df),
        topics=_export_topics(df, top_n=15),
        weekly_by_cat=_export_weekly_by_cat(df),
        subcategories=[],  # populated only when LLM is connected
        filter_summary=_filter_summary(date_from, date_to, category, severity, lang),
        lang=lang,
    )
    return _xlsx_response(data, report="patterns", lang=lang)


@app.get("/api/export/recommendations")
def export_recommendations(
    snapshot_id: Optional[int] = None,
    lang: str = Query(default="ar"),
):
    """Export the named snapshot, or the most recent one if `snapshot_id` is omitted."""
    snap: Optional[dict] = None
    if snapshot_id is not None:
        snap = recs.get_snapshot(snapshot_id)
    else:
        runs = recs.list_snapshots(limit=1)
        if runs:
            snap = recs.get_snapshot(runs[0]["id"])

    insights: list[dict] = []
    kpis_d: dict = {}
    if snap:
        insights = list(snap.get("insights") or [])
        kpis_d   = dict(snap.get("kpis") or {})

    data = excel_export.build_recommendations_workbook(
        snapshot=snap, insights=insights, kpis=kpis_d, lang=lang,
    )
    return _xlsx_response(data, report="recommendations", lang=lang)


@app.get("/api/export/tickets")
def export_tickets(
    date_from: Optional[date] = None,
    date_to: Optional[date]   = None,
    category: Optional[list[str]] = Query(default=None),
    severity: Optional[list[str]] = Query(default=None),
    lang: str = Query(default="ar"),
):
    df = _params(date_from, date_to, category, severity)

    # Cap export size — 50k rows is plenty and keeps file under ~10 MB.
    if df is not None and len(df) > 50_000:
        df = df.head(50_000)

    data = excel_export.build_tickets_workbook(
        df=df,
        filter_summary=_filter_summary(date_from, date_to, category, severity, lang),
        lang=lang,
    )
    return _xlsx_response(data, report="tickets", lang=lang)
