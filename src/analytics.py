"""Pure analytics — clustering, KPIs, trend rollups, anomaly detection.

Used by both prepare_data.py (offline cache) and app.py (online slicing).
No external services. Deterministic given the same input.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

log = logging.getLogger(__name__)

# Arabic stop-words list — small but practical for this domain.
ARABIC_STOPWORDS = {
    "في", "من", "على", "إلى", "عن", "هذا", "هذه", "ذلك", "تلك",
    "ما", "هل", "ثم", "قد", "لقد", "بعد", "قبل", "كل", "بعض",
    "غير", "حتى", "حيث", "بين", "كما", "هو", "هي", "هم", "نحن",
    "أن", "إن", "لا", "لم", "لن", "ليس", "هنا", "هناك",
    "أو", "و", "ل", "ب", "ك",
}

SEVERITY_ORDER = ["عالية", "متوسطة", "منخفضة"]


# ---------- Topic clustering ----------

def assign_topic_clusters(
    df: pd.DataFrame,
    *,
    text_col: str = "body",
    category_col: str = "category",
    k: int = 8,
    random_state: int = 42,
) -> pd.DataFrame:
    """Add `topic_cluster_id` and `topic_cluster_terms` to df.

    Uses TF-IDF over (category + body) and KMeans. Determines per-cluster
    top terms for human labelling. Operates in-place-style but returns df.
    """
    out = df.copy()
    corpus = (out[category_col].fillna("") + " " + out[text_col].fillna("")).str.strip()
    if corpus.str.strip().eq("").all():
        out["topic_cluster_id"] = -1
        out["topic_cluster_terms"] = ""
        return out

    vec = TfidfVectorizer(
        max_features=2000,
        ngram_range=(1, 2),
        min_df=2,
        stop_words=list(ARABIC_STOPWORDS),
    )
    X = vec.fit_transform(corpus)
    n_clusters = max(2, min(k, X.shape[0] - 1))
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(X)

    # Top 4 terms per cluster (for human-readable label)
    terms = np.array(vec.get_feature_names_out())
    top_terms = []
    for ci in range(n_clusters):
        order = km.cluster_centers_[ci].argsort()[::-1][:4]
        top_terms.append("، ".join(terms[order]))

    out["topic_cluster_id"] = labels
    out["topic_cluster_terms"] = [top_terms[lbl] for lbl in labels]
    return out


# ---------- KPI rollups ----------

@dataclass
class KPIs:
    total: int
    by_category: pd.Series
    by_severity: pd.Series
    weekly_volume: pd.Series          # week_start -> count
    pct_high_severity: float
    pct_complaints: float


def compute_kpis(df: pd.DataFrame) -> KPIs:
    if df.empty:
        empty = pd.Series(dtype="int64")
        return KPIs(
            total=0,
            by_category=empty,
            by_severity=empty,
            weekly_volume=pd.Series(dtype="int64"),
            pct_high_severity=0.0,
            pct_complaints=0.0,
        )
    by_cat = df["category"].value_counts()
    by_sev = (
        df["severity"]
        .value_counts()
        .reindex(SEVERITY_ORDER, fill_value=0)
    )
    weekly = (
        df.assign(week=pd.to_datetime(df["week_start"]))
        .groupby("week")
        .size()
        .sort_index()
    )
    pct_high = float(by_sev.get("عالية", 0)) / len(df) * 100.0
    pct_compl = float((df["category"] == "شكوى").sum()) / len(df) * 100.0
    return KPIs(
        total=len(df),
        by_category=by_cat,
        by_severity=by_sev,
        weekly_volume=weekly,
        pct_high_severity=pct_high,
        pct_complaints=pct_compl,
    )


# ---------- Recurring patterns ----------

def top_recurring_topics(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Top topic_label values with volume and severity mix."""
    if df.empty:
        return pd.DataFrame(
            columns=["topic_label", "count", "share_pct", "high_pct", "sample_body"]
        )
    grp = df.groupby("topic_label", dropna=False)
    high_pct = (
        grp["severity"]
        .apply(lambda s: 100.0 * (s == "عالية").mean())
    )
    out = pd.DataFrame({
        "count": grp.size(),
        "high_pct": high_pct,
        "sample_body": grp["body"].first(),
    })
    out["share_pct"] = 100.0 * out["count"] / len(df)
    out = out.sort_values("count", ascending=False).head(top_n).reset_index()
    return out[["topic_label", "count", "share_pct", "high_pct", "sample_body"]]


# ---------- Anomaly / early-warning ----------

@dataclass
class AnomalyAlert:
    week_start: pd.Timestamp
    dimension: str           # "category" or "topic_label"
    value: str               # e.g. "شكوى"
    count: int
    baseline_mean: float
    baseline_std: float
    z_score: float
    severity: str            # "عالية" | "متوسطة"
    suggested_action: str


def detect_weekly_anomalies(
    df: pd.DataFrame,
    *,
    z_threshold: float = 1.8,
    min_count: int = 5,
) -> list[AnomalyAlert]:
    """Flag categories/topics whose latest-week volume jumps vs. their own baseline."""
    if df.empty or "week_start" not in df.columns:
        return []

    df = df.copy()
    df["week_start"] = pd.to_datetime(df["week_start"])
    weeks = sorted(df["week_start"].dropna().unique())
    if len(weeks) < 3:
        return []
    latest = weeks[-1]
    history = weeks[:-1]

    alerts: list[AnomalyAlert] = []

    for dim in ("category", "topic_label"):
        if dim not in df.columns:
            continue
        # weekly counts per (week, value)
        wide = (
            df.groupby(["week_start", dim])
            .size()
            .unstack(fill_value=0)
            .sort_index()
        )
        if latest not in wide.index:
            continue
        latest_row = wide.loc[latest]
        hist = wide.loc[wide.index.isin(history)]

        for value, count in latest_row.items():
            if count < min_count:
                continue
            series = hist[value] if value in hist.columns else pd.Series([0])
            mu = float(series.mean()) if len(series) else 0.0
            sd = float(series.std(ddof=0)) if len(series) > 1 else 0.0
            if sd == 0:
                # Cold start: flag if more than 2x the historical max
                hist_max = float(series.max()) if len(series) else 0.0
                if count > max(hist_max * 2, min_count):
                    z = float("inf")
                else:
                    continue
            else:
                z = (count - mu) / sd
            if z >= z_threshold or z == float("inf"):
                sev = "عالية" if (z >= 3 or z == float("inf")) else "متوسطة"
                action = _action_for_anomaly(dim, value, count, mu, sev)
                alerts.append(
                    AnomalyAlert(
                        week_start=pd.Timestamp(latest),
                        dimension=dim,
                        value=str(value),
                        count=int(count),
                        baseline_mean=mu,
                        baseline_std=sd,
                        z_score=float(z if z != float("inf") else 99.0),
                        severity=sev,
                        suggested_action=action,
                    )
                )

    # Most extreme first
    alerts.sort(key=lambda a: (-a.z_score, -a.count))
    return alerts


def _action_for_anomaly(dim: str, value: str, count: int,
                         baseline: float, severity: str) -> str:
    delta = count - baseline
    if dim == "category":
        if value == "شكوى":
            return (
                f"ارتفاع غير معتاد في الشكاوى هذا الأسبوع (+{delta:.0f} "
                f"عن المعدل). يُوصى بمراجعة جذور الشكاوى وإطلاق حملة تواصل استباقي."
            )
        if value == "دعم فني":
            return (
                f"تصاعد طلبات الدعم الفني (+{delta:.0f}). يُوصى بفحص توافر "
                f"الخدمات الرقمية وحالة القنوات الإلكترونية."
            )
        return (
            f"ارتفاع غير معتاد في فئة «{value}» (+{delta:.0f}). "
            f"يُوصى بمراجعة محتوى الفئة وإجراء تحليل الأسباب الجذرية."
        )
    # topic dim
    return (
        f"ارتفاع غير معتاد في موضوع «{value}» (+{delta:.0f} عن المعدل). "
        f"يُقترح تحسين استباقي على الخدمة المرتبطة وتحديث قاعدة المعرفة."
    )


# ---------- Free-text context summary (for the Q&A page) ----------

def forecast_weekly(df: pd.DataFrame,
                    horizon: int = 2,
                    by_category: bool = False) -> dict:
    """Weighted-average forecast for the next `horizon` weeks.

    Method: simple exponential smoothing with α=0.4 over the recent weekly
    series, plus a ±1σ band derived from residuals against a 4-week moving
    average. Designed to be honest about uncertainty for short series.

    Returns dict:
        {
          "history": {"x": [...], "y": [...]},
          "forecast": {"x": [...], "y": [...], "lo": [...], "hi": [...]},
          "by_category": [ { "name": "...", "x": [...], "y": [...] } ] (optional)
        }
    """
    if df.empty or "week_start" not in df.columns:
        return {"history": {"x": [], "y": []}, "forecast": {"x": [], "y": [], "lo": [], "hi": []}}

    df = df.copy()
    df["_w"] = pd.to_datetime(df["week_start"])
    weekly = df.groupby("_w").size().sort_index()
    if len(weekly) < 4:
        return {
            "history": {
                "x": [d.strftime("%Y-%m-%d") for d in weekly.index],
                "y": [int(v) for v in weekly.values],
            },
            "forecast": {"x": [], "y": [], "lo": [], "hi": []},
        }

    # exponential smoothing
    alpha = 0.4
    series = weekly.values.astype(float)
    s = series[0]
    smoothed = [s]
    for v in series[1:]:
        s = alpha * v + (1 - alpha) * s
        smoothed.append(s)
    last_smoothed = smoothed[-1]

    # confidence band from residuals against 4-wk MA
    ma = pd.Series(series).rolling(window=4, min_periods=2).mean().values
    resid = series - ma
    sigma = float(np.nanstd(resid[~np.isnan(resid)])) if np.any(~np.isnan(resid)) else 0.0

    last_date = weekly.index[-1]
    fx, fy, flo, fhi = [], [], [], []
    for i in range(1, horizon + 1):
        d = (last_date + pd.Timedelta(weeks=i))
        fx.append(d.strftime("%Y-%m-%d"))
        # widen the band slightly per step ahead
        widen = 1 + 0.25 * (i - 1)
        y = max(0, last_smoothed)
        fy.append(round(y, 1))
        flo.append(max(0, round(y - sigma * widen, 1)))
        fhi.append(round(y + sigma * widen, 1))

    out = {
        "history": {
            "x": [d.strftime("%Y-%m-%d") for d in weekly.index],
            "y": [int(v) for v in weekly.values],
        },
        "forecast": {"x": fx, "y": fy, "lo": flo, "hi": fhi},
    }

    if by_category and "category" in df.columns:
        cat_series = (
            df.groupby(["_w", "category"]).size().unstack(fill_value=0).sort_index()
        )
        cats = []
        for cat in cat_series.columns:
            v = cat_series[cat].values.astype(float)
            if len(v) < 4 or v.sum() < 5:
                continue
            ss = v[0]
            for x in v[1:]:
                ss = alpha * x + (1 - alpha) * ss
            yhat = max(0, round(ss, 1))
            cats.append({
                "name": cat,
                "next_week": yhat,
                "last_week": int(v[-1]),
                "delta": round(yhat - v[-1], 1),
            })
        cats.sort(key=lambda c: -c["next_week"])
        out["by_category"] = cats

    return out


_AR_NORMALIZE = str.maketrans({
    "أ": "ا", "إ": "ا", "آ": "ا",
    "ى": "ي", "ة": "ه",
    "ـ": "",
})


def _normalize_for_dedup(text: str) -> str:
    s = (text or "").strip().lower().translate(_AR_NORMALIZE)
    # collapse whitespace and strip diacritics-like punctuation
    s = " ".join(s.split())
    return s


def find_recurring_cases(df: pd.DataFrame,
                         min_repeats: int = 3,
                         lookback_days: int = 60) -> pd.DataFrame:
    """Cluster near-identical request bodies recurring within `lookback_days`.

    With no beneficiary_id in the source, this is the closest signal to
    'repeat-beneficiary' detection — identifying request bodies that recur,
    which is either the same beneficiary contacting again or a systemic
    issue affecting many. Both are worth surfacing.

    Returns columns: phrase, count, first_seen, last_seen, top_category,
                     high_pct, sample_ids.
    """
    if df.empty or "body" not in df.columns:
        return pd.DataFrame(columns=["phrase", "count", "first_seen", "last_seen",
                                     "top_category", "high_pct", "sample_ids"])

    cutoff = pd.to_datetime(df["closed_at"]).max() - pd.Timedelta(days=lookback_days)
    recent = df[pd.to_datetime(df["closed_at"]) >= cutoff].copy()
    if recent.empty:
        return pd.DataFrame(columns=["phrase", "count", "first_seen", "last_seen",
                                     "top_category", "high_pct", "sample_ids"])

    recent["_norm"] = recent["body"].astype(str).apply(_normalize_for_dedup)
    recent = recent[recent["_norm"].str.len() >= 4]
    grouped = recent.groupby("_norm")
    rows = []
    for norm, g in grouped:
        n = len(g)
        if n < min_repeats:
            continue
        rows.append({
            "phrase": g["body"].iloc[0],
            "count": int(n),
            "first_seen": pd.to_datetime(g["closed_at"]).min().strftime("%Y-%m-%d"),
            "last_seen":  pd.to_datetime(g["closed_at"]).max().strftime("%Y-%m-%d"),
            "top_category": g["category"].mode().iloc[0],
            "high_pct": round(100.0 * (g["severity"] == "عالية").mean(), 1),
            "sample_ids": g["request_id"].head(5).astype(int).tolist(),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["count", "high_pct"], ascending=[False, False]).reset_index(drop=True)


def severity_by_week(df: pd.DataFrame) -> pd.DataFrame:
    """Stacked-area-friendly weekly severity rollup."""
    if df.empty:
        return pd.DataFrame(columns=["week_start"] + SEVERITY_ORDER)
    df = df.assign(_w=pd.to_datetime(df["week_start"]))
    pivot = (
        df.groupby(["_w", "severity"])
          .size()
          .unstack(fill_value=0)
          .reindex(columns=SEVERITY_ORDER, fill_value=0)
          .sort_index()
          .reset_index()
          .rename(columns={"_w": "week_start"})
    )
    return pivot


def category_severity_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Counts of records per (category, severity)."""
    if df.empty:
        return pd.DataFrame(columns=["category"] + SEVERITY_ORDER)
    pivot = (
        df.groupby(["category", "severity"])
          .size()
          .unstack(fill_value=0)
          .reindex(columns=SEVERITY_ORDER, fill_value=0)
          .reset_index()
    )
    return pivot


def topic_momentum(df: pd.DataFrame, lookback_weeks: int = 4, top_n: int = 6) -> pd.DataFrame:
    """Topics whose volume in the last `lookback_weeks` grew the most vs the
    earlier period of the same length."""
    if df.empty or "week_start" not in df.columns:
        return pd.DataFrame(columns=["topic_label", "recent", "prior", "delta", "growth_pct"])
    weeks = sorted(pd.to_datetime(df["week_start"]).unique())
    if len(weeks) < lookback_weeks * 2:
        return pd.DataFrame(columns=["topic_label", "recent", "prior", "delta", "growth_pct"])
    cutoff = weeks[-lookback_weeks]
    prior_start = weeks[-lookback_weeks * 2]
    df = df.assign(_w=pd.to_datetime(df["week_start"]))
    recent = df[df["_w"] >= cutoff]["topic_label"].value_counts()
    prior = df[(df["_w"] >= prior_start) & (df["_w"] < cutoff)]["topic_label"].value_counts()
    topics = sorted(set(recent.index) | set(prior.index))
    rows = []
    for t in topics:
        r = int(recent.get(t, 0))
        p = int(prior.get(t, 0))
        if r + p < 5:
            continue
        delta = r - p
        growth = ((r - p) / p * 100.0) if p > 0 else (100.0 if r > 0 else 0.0)
        rows.append({"topic_label": t, "recent": r, "prior": p, "delta": delta, "growth_pct": growth})
    out = pd.DataFrame(rows).sort_values("growth_pct", ascending=False).head(top_n)
    return out.reset_index(drop=True)


def build_signals_text(df: pd.DataFrame) -> str:
    """Compact, factual description of the dataset for LLM consumption."""
    if df.empty:
        return "لا توجد بيانات ضمن النطاق المحدد."
    k = compute_kpis(df)
    top_topics = top_recurring_topics(df, top_n=6)
    momentum = topic_momentum(df)
    alerts = detect_weekly_anomalies(df)
    lines = [
        f"إجمالي السجلات: {k.total} · شكاوى {k.pct_complaints:.0f}٪ · "
        f"خطورة عالية {k.pct_high_severity:.0f}٪",
        "",
        "الفئات وحجمها:",
    ]
    for c, n in k.by_category.items():
        lines.append(f"  - {c}: {n} طلب")
    lines.append("")
    lines.append("أبرز الموضوعات المتكررة:")
    for _, r in top_topics.iterrows():
        lines.append(
            f"  - {r['topic_label']}: {r['count']} طلب "
            f"(خطورة عالية {r['high_pct']:.0f}٪)"
        )
    if not momentum.empty:
        lines.append("")
        lines.append("موضوعات متصاعدة (آخر ٤ أسابيع مقارنة بما قبلها):")
        for _, r in momentum.iterrows():
            arrow = "↑" if r["growth_pct"] > 0 else "↓"
            lines.append(
                f"  - {r['topic_label']}: {r['recent']} مقابل {r['prior']} "
                f"({arrow}{abs(r['growth_pct']):.0f}٪)"
            )
    if alerts:
        lines.append("")
        lines.append("ارتفاعات شاذة في آخر أسبوع:")
        for a in alerts[:5]:
            lines.append(
                f"  - {a.value} ({a.dimension}): {a.count} طلب "
                f"(متوسط تاريخي {a.baseline_mean:.1f})"
            )
    return "\n".join(lines)


def _topic_action_map(topic: str) -> tuple[str, str]:
    """(responsible_unit, action) mapping per topic for financial-services CX."""
    t = (topic or "")
    if "تعثّر" in t or "تعثر" in t or "السداد" in t:
        return ("فريق التحصيل وإدارة الائتمان",
                "مراجعة محفظة المتأثرين وتفعيل خطط جدولة سداد مرنة، "
                "والتواصل الاستباقي بمستشار مالي مخصّص.")
    if "احتيال" in t:
        return ("إدارة المخاطر والامتثال",
                "تفعيل بروتوكول الكشف عن الاحتيال على المعاملات المرتبطة "
                "وإيقاف أي عمليات معلّقة لحين التحقق.")
    if "خصم" in t:
        return ("إدارة العمليات المالية",
                "مراجعة سجل الخصم لكل طلب، وإجراء التسويات في الحالات "
                "الموثّقة خلال 48 ساعة.")
    if "قرض" in t or "تمويل" in t:
        return ("إدارة الائتمان",
                "مراجعة سياسة الاستجابة لطلبات التمويل، وتقصير دورة الاعتماد "
                "للحالات الواضحة.")
    if "دعم" in t:
        return ("إدارة برامج الدعم",
                "تحديث مرجعية المعايير على البوابة، وإطلاق دفعة تواصل "
                "استباقية للمستفيدين المؤهلين.")
    if "تحديث" in t and "بيانات" in t:
        return ("التحول الرقمي",
                "تفعيل خدمة التحديث الذاتي عبر القناة الرقمية لخفض "
                "الطلبات اليدوية.")
    if "قنوات" in t or "الدعم الفني" in t:
        return ("إدارة التحول الرقمي والدعم الفني",
                "مراجعة لوحة توافر الخدمات الرقمية وتحسين رحلة المستخدم "
                "في المسارات الأكثر إثارة للطلبات.")
    if "خدمة" in t and "مراجع" in t:
        return ("مكتب خدمة المراجعين",
                "مراجعة قائمة المتطلبات الناقصة الأكثر تكراراً وتبسيط النموذج.")
    if "شكاو" in t:
        return ("إدارة تجربة المستفيد",
                "تشكيل فريق سريع لتحليل أسباب الشكاوى المتكررة هذا الأسبوع "
                "وإطلاق إصلاحات قصيرة المدى.")
    return ("الجهة المختصة بالموضوع",
            "إحالة المسار إلى الجهة المعنية مع SLA لا يتجاوز 3 أيام عمل.")


def rule_based_insights(df: pd.DataFrame) -> list[dict]:
    """Deterministic insights for financial-services CX when no LLM is available.

    Each insight: {title, evidence, action, metric}.
    """
    out: list[dict] = []
    if df.empty:
        return out

    k = compute_kpis(df)

    # 1) Anomalies — most actionable first
    alerts = detect_weekly_anomalies(df)
    if alerts:
        a = alerts[0]
        unit, _ = _topic_action_map(a.value)
        out.append({
            "title": f"ارتفاع غير معتاد في «{a.value}» خلال آخر أسبوع",
            "evidence": (
                f"رصد النظام {a.count} طلب في آخر أسبوع مقابل متوسط تاريخي "
                f"{a.baseline_mean:.0f} (انحراف معياري {a.z_score:.1f}σ)."
            ),
            "action": f"تكليف ({unit}) بمراجعة الموجة فوراً، {a.suggested_action}",
            "metric": "عودة الحجم الأسبوعي إلى المتوسط التاريخي خلال أسبوعين.",
        })

    # 2) Topic with high-severity concentration
    topics = top_recurring_topics(df, top_n=10)
    bad = topics[(topics["high_pct"] >= 50) & (topics["count"] >= 30)]
    if not bad.empty:
        r = bad.iloc[0]
        unit, action_body = _topic_action_map(r["topic_label"])
        out.append({
            "title": f"موضوع «{r['topic_label']}» يستنزف موارد الفريق بخطورة عالية",
            "evidence": (
                f"{int(r['count'])} طلب في الفترة، منها {r['high_pct']:.0f}٪ "
                f"بخطورة عالية — أعلى تركيز خطورة بين الموضوعات."
            ),
            "action": f"إحالة المسار إلى ({unit}). {action_body}",
            "metric": f"خفض نسبة الخطورة العالية في الموضوع إلى ما دون 30٪ خلال 6 أسابيع.",
        })

    # 3) Topic momentum (rising)
    mom = topic_momentum(df, lookback_weeks=4, top_n=5)
    rising = mom[mom["growth_pct"] > 30]
    if not rising.empty:
        r = rising.iloc[0]
        unit, action_body = _topic_action_map(r["topic_label"])
        out.append({
            "title": f"تصاعد متسارع في موضوع «{r['topic_label']}»",
            "evidence": (
                f"الطلبات ارتفعت من {int(r['prior'])} إلى {int(r['recent'])} "
                f"(+{r['growth_pct']:.0f}٪) في آخر 4 أسابيع."
            ),
            "action": f"({unit}): {action_body}",
            "metric": "خفض الطلبات في هذا الموضوع بنسبة 30٪ خلال شهر.",
        })

    # 4) Complaint rate
    if k.pct_complaints >= 15:
        top_complaint_topic = (
            df[df["category"] == "شكوى"]["topic_label"].value_counts().head(1)
        )
        topic_name = top_complaint_topic.index[0] if not top_complaint_topic.empty else "—"
        topic_count = int(top_complaint_topic.iloc[0]) if not top_complaint_topic.empty else 0
        out.append({
            "title": "ارتفاع نسبة الشكاوى يستحق إصلاحاً منهجياً",
            "evidence": (
                f"الشكاوى تمثّل {k.pct_complaints:.0f}٪ من إجمالي الطلبات. "
                f"أكبر مساهم: «{topic_name}» بـ{topic_count} شكوى."
            ),
            "action": (
                "تشكيل فريق من إدارة تجربة المستفيد + إدارة العمليات لمراجعة "
                "أسباب الشكاوى المرتبطة بـ«" + topic_name + "» وإطلاق إصلاحات سريعة."
            ),
            "metric": "خفض نسبة الشكاوى الكلية إلى أقل من 12٪ خلال شهرين.",
        })

    # 5) Beneficiary self-service opportunity
    info_topics = topics[topics["topic_label"].astype(str).str.contains(
        "استعلام|تحديث|الاستفسار", regex=True, na=False
    )]
    info_total = int(info_topics["count"].sum()) if not info_topics.empty else 0
    if info_total >= 200:
        share = info_total / k.total * 100
        out.append({
            "title": "فرصة لخفض الطلبات الإدارية عبر الخدمة الذاتية",
            "evidence": (
                f"الطلبات الإدارية (استعلامات وتحديث بيانات) تمثّل {share:.0f}٪ "
                f"من الإجمالي ({info_total} طلب)."
            ),
            "action": (
                "تفعيل واجهة خدمة ذاتية للاستعلامات الأكثر تكراراً عبر بوابة "
                "الصندوق وتطبيق الجوال، مع روابط مباشرة من رسائل الإشعارات."
            ),
            "metric": "خفض حجم هذه الطلبات في القناة البشرية بنسبة 25٪ خلال 8 أسابيع.",
        })

    return out[:5]


def summarize_for_qa(df: pd.DataFrame, max_topics: int = 6) -> str:
    if df.empty:
        return "لا توجد بيانات ضمن النطاق المحدد."
    k = compute_kpis(df)
    top_topics = top_recurring_topics(df, top_n=max_topics)
    lines = [
        f"إجمالي السجلات: {k.total}",
        f"نسبة الشكاوى: {k.pct_complaints:.1f}%",
        f"نسبة الخطورة العالية: {k.pct_high_severity:.1f}%",
        "التوزيع حسب الفئة:",
    ]
    for cat, n in k.by_category.items():
        lines.append(f"  - {cat}: {n}")
    lines.append("التوزيع حسب الخطورة:")
    for sev, n in k.by_severity.items():
        lines.append(f"  - {sev}: {n}")
    lines.append("أبرز المواضيع المتكررة:")
    for _, r in top_topics.iterrows():
        lines.append(
            f"  - {r['topic_label']}: {r['count']} طلب "
            f"({r['share_pct']:.1f}% من الإجمالي، {r['high_pct']:.0f}% خطورة عالية)"
        )
    return "\n".join(lines)
