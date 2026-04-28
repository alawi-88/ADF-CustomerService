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
    z_threshold: float = 1.3,
    min_count: int = 5,
    scan_recent_weeks: int = 4,
) -> list[AnomalyAlert]:
    """Detect anomalies across the recent weeks (not just the latest, which is
    often incomplete). Catches spikes AND sustained elevations.

    Strategy:
      • For each (category, topic), compute weekly counts.
      • Treat the LAST FULL week as week-of-interest (skip the latest if it
        appears mid-week / smaller than every prior).
      • Compare its count against the prior baseline (mean+std over the rest).
      • Also flag SUSTAINED elevation: count above baseline for ≥2 of last 3 weeks.
    """
    if df.empty or "week_start" not in df.columns:
        return []

    df = df.copy()
    df["week_start"] = pd.to_datetime(df["week_start"])
    weeks = sorted(df["week_start"].dropna().unique())
    if len(weeks) < 4:
        return []

    # Skip the very-latest week if it looks partial (count noticeably lower than
    # the full series average — typical when data ends mid-week).
    weekly_total = df.groupby("week_start").size().sort_index()
    if len(weekly_total) >= 4:
        latest_total = float(weekly_total.iloc[-1])
        prior_avg = float(weekly_total.iloc[:-1].mean())
        if latest_total < prior_avg * 0.4:
            weeks = weeks[:-1]
    if len(weeks) < 3:
        return []
    target_week = weeks[-1]
    history_weeks = weeks[:-1]

    alerts: list[AnomalyAlert] = []
    seen: set[tuple[str, str]] = set()

    for dim in ("category", "topic_label"):
        if dim not in df.columns:
            continue
        wide = (
            df.groupby(["week_start", dim]).size().unstack(fill_value=0).sort_index()
        )
        if target_week not in wide.index:
            continue
        target_row = wide.loc[target_week]
        hist = wide.loc[wide.index.isin(history_weeks)]

        for value, count in target_row.items():
            if count < min_count:
                continue
            series = hist[value] if value in hist.columns else pd.Series([0])
            mu = float(series.mean()) if len(series) else 0.0
            sd = float(series.std(ddof=0)) if len(series) > 1 else 0.0

            # 1) z-score spike
            z = 0.0
            if sd == 0:
                hist_max = float(series.max()) if len(series) else 0.0
                if count > max(hist_max * 1.6, min_count):
                    z = 99.0
            else:
                z = (count - mu) / sd

            if z >= z_threshold:
                sev = "عالية" if z >= 2.5 else "متوسطة"
                key = (dim, str(value))
                if key not in seen:
                    seen.add(key)
                    alerts.append(AnomalyAlert(
                        week_start=pd.Timestamp(target_week),
                        dimension=dim, value=str(value), count=int(count),
                        baseline_mean=mu, baseline_std=sd,
                        z_score=float(z),
                        severity=sev,
                        suggested_action=_action_for_anomaly(dim, value, count, mu, sev),
                    ))
                continue

            # 2) sustained elevation: 2+ of last 3 weeks above baseline
            last3_keys = weeks[-min(3, len(weeks)):]
            if value in wide.columns:
                last3 = wide.loc[wide.index.isin(last3_keys), value].values
                if len(last3) >= 3 and mu > 0:
                    elevated = sum(1 for v in last3 if v > mu * 1.25 and v >= min_count)
                    if elevated >= 2:
                        key = (dim, str(value))
                        if key not in seen:
                            seen.add(key)
                            alerts.append(AnomalyAlert(
                                week_start=pd.Timestamp(target_week),
                                dimension=dim, value=str(value), count=int(count),
                                baseline_mean=mu, baseline_std=sd,
                                z_score=1.0,
                                severity="متوسطة",
                                suggested_action=(
                                    f"ارتفاع متواصل في «{value}» على مدى أسابيع عدة "
                                    "(مستوى أعلى من المتوسط بـ 25٪ أو أكثر). "
                                    + _action_for_anomaly(dim, value, count, mu, "متوسطة")
                                ),
                            ))

    # 3) New-emergence: topic that wasn't present in earlier history but is now
    if "topic_label" in df.columns and len(weeks) >= 6:
        early = set(df[df["week_start"].isin(weeks[:len(weeks)//2])]["topic_label"].unique())
        recent_subset = df[df["week_start"].isin(weeks[-3:])]
        recent_counts = recent_subset["topic_label"].value_counts()
        for topic, count in recent_counts.items():
            if topic in early or count < min_count * 2:
                continue
            key = ("topic_label", str(topic))
            if key in seen:
                continue
            seen.add(key)
            alerts.append(AnomalyAlert(
                week_start=pd.Timestamp(target_week),
                dimension="topic_label", value=str(topic), count=int(count),
                baseline_mean=0.0, baseline_std=0.0, z_score=99.0,
                severity="متوسطة",
                suggested_action=(
                    f"موضوع جديد ظهر مؤخراً «{topic}» بحجم {int(count)} طلب — "
                    "يستحق فهم سببه قبل أن يتوسّع."
                ),
            ))

    alerts.sort(key=lambda a: (-a.z_score, -a.count))

    # 4) FALLBACK — if no spike-based alerts surface, identify the top
    # high-severity hotspots (categories/topics whose high-severity rate
    # is significantly above the dataset average). These are always
    # actionable signals worth monitoring even without a recent spike.
    if not alerts:
        avg_high_pct = float((df["severity"] == "عالية").mean() * 100.0)
        # category hotspots
        cat_stats = df.groupby("category").agg(
            count=("severity", "size"),
            high_count=("severity", lambda s: int((s == "عالية").sum())),
        )
        cat_stats["high_pct"] = cat_stats["high_count"] / cat_stats["count"] * 100.0
        cat_hot = cat_stats[(cat_stats["high_pct"] >= max(avg_high_pct * 1.5, 30))
                            & (cat_stats["count"] >= min_count * 4)]
        for cat, row in cat_hot.iterrows():
            alerts.append(AnomalyAlert(
                week_start=pd.Timestamp(target_week),
                dimension="category", value=str(cat), count=int(row["count"]),
                baseline_mean=avg_high_pct, baseline_std=0.0, z_score=1.0,
                severity="عالية" if row["high_pct"] >= 60 else "متوسطة",
                suggested_action=(
                    f"بؤرة خطورة عالية: «{cat}» تسجل {row['high_pct']:.0f}٪ "
                    f"من طلباتها بخطورة عالية (مقابل متوسط {avg_high_pct:.0f}٪ "
                    "في كامل البيانات). مراجعة جذور هذه الفئة وإطلاق إصلاحات سريعة."
                ),
            ))
        # topic hotspots (high-volume + high-severity)
        if "topic_label" in df.columns:
            top = top_recurring_topics(df, top_n=15)
            top_hot = top[(top["high_pct"] >= max(avg_high_pct * 1.5, 35))
                          & (top["count"] >= min_count * 4)].head(5)
            for _, row in top_hot.iterrows():
                alerts.append(AnomalyAlert(
                    week_start=pd.Timestamp(target_week),
                    dimension="topic_label", value=str(row["topic_label"]),
                    count=int(row["count"]),
                    baseline_mean=avg_high_pct, baseline_std=0.0, z_score=1.0,
                    severity="عالية" if row["high_pct"] >= 70 else "متوسطة",
                    suggested_action=(
                        f"تركّز الخطورة في «{row['topic_label']}» — "
                        f"{row['high_pct']:.0f}٪ من {int(row['count'])} طلب "
                        "بخطورة عالية. مراجعة سياسة الاستجابة لهذا الموضوع "
                        "وتفعيل قناة سريعة لمعالجته."
                    ),
                ))
        alerts.sort(key=lambda a: -a.count)

    return alerts[:15]


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


def _examples_for(df: pd.DataFrame, n: int = 3, **filters) -> list[dict]:
    """Pick representative examples (id + body excerpt) matching filters."""
    sl = df
    for col, val in filters.items():
        if col == "topic_label":
            sl = sl[sl["topic_label"] == val]
        elif col == "category":
            sl = sl[sl["category"] == val]
        elif col == "severity":
            sl = sl[sl["severity"] == val]
    sl = sl.sort_values("closed_at", ascending=False).head(n)
    out = []
    for _, r in sl.iterrows():
        body = str(r.get("body") or "").strip()
        if len(body) > 70:
            body = body[:70].rstrip() + "…"
        out.append({"id": int(r["request_id"]), "body": body})
    return out


def _attribute_causes(df: pd.DataFrame, *, category: str | None = None,
                      topic: str | None = None) -> list[dict]:
    """Decompose 'why is this X concentrated/elevated?' into ranked causes.

    Returns up to 3 hypothesised drivers with a probability % and a reason
    text grounded in the data signals. The probabilities are not Bayesian —
    they are share-of-evidence weights normalised to 100%, surfacing the
    Pareto pattern the user explicitly asked for.
    """
    if df.empty:
        return []

    # Build the slice we are explaining
    sl = df
    if category: sl = sl[sl["category"] == category]
    if topic:    sl = sl[sl["topic_label"] == topic]
    if sl.empty:
        return []

    candidates: list[dict] = []

    # Cause 1: dominant body phrase (recurring text)
    if "body" in sl.columns:
        norm = sl["body"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
        top_phrase = norm.value_counts().head(1)
        if not top_phrase.empty:
            phrase, n = top_phrase.index[0], int(top_phrase.iloc[0])
            share = n / len(sl)
            if n >= 3 and share >= 0.05:
                candidates.append({
                    "cause": f"تكرار النص ذاته «{phrase[:60]}»",
                    "weight": share * 0.9,
                    "evidence": f"{n} طلب من أصل {len(sl)} يحملون النص نفسه — يرجّح أن المصدر "
                                "مستفيد متكرر أو مشكلة منهجية واحدة.",
                })

    # Cause 2: dominant topic within the slice
    if topic is None and "topic_label" in sl.columns:
        top_topic = sl["topic_label"].value_counts().head(1)
        if not top_topic.empty:
            tname, tn = top_topic.index[0], int(top_topic.iloc[0])
            tshare = tn / len(sl)
            if tshare >= 0.20:
                candidates.append({
                    "cause": f"تركّز في موضوع «{tname}»",
                    "weight": tshare * 0.85,
                    "evidence": f"{tn} طلب ({tshare*100:.0f}٪ من الشريحة) ينتمون لموضوع واحد — "
                                "يرجّح أن المصدر مشكلة محدّدة في هذا المسار.",
                })

    # Cause 3: dominant category within the slice (when explaining a topic)
    if category is None and "category" in sl.columns:
        top_cat = sl["category"].value_counts().head(1)
        if not top_cat.empty:
            cn = int(top_cat.iloc[0])
            cshare = cn / len(sl)
            if cshare >= 0.50:
                candidates.append({
                    "cause": f"كل الطلبات تقريباً في فئة «{top_cat.index[0]}»",
                    "weight": cshare * 0.7,
                    "evidence": f"{cshare*100:.0f}٪ من الشريحة من فئة واحدة — التحدي تشغيلي "
                                "في فريق هذه الفئة وليس عاماً.",
                })

    # Cause 4: high-severity signal keyword presence
    sev_kw = ["متعسر", "تعثر", "متأخر", "تأخر", "إيقاف", "ايقاف", "خصم", "احتيال",
              "رفض", "استرداد", "مستحقات"]
    body_text = sl["body"].astype(str).str.cat(sep=" ")
    hits = []
    for kw in sev_kw:
        c = body_text.count(kw)
        if c >= 3:
            hits.append((kw, c))
    if hits:
        hits.sort(key=lambda x: -x[1])
        top_kw, top_c = hits[0]
        share = top_c / len(sl)
        candidates.append({
            "cause": f"إشارات حساسة في النص (مثل «{top_kw}»)",
            "weight": min(0.6, share * 0.6),
            "evidence": f"كلمة «{top_kw}» وردت {top_c} مرة في الشريحة — تشير إلى "
                        "مشكلة مالية/تشغيلية تستدعي التصعيد.",
        })

    # Cause 5: temporal concentration — same week or day
    if "closed_at" in sl.columns and len(sl) >= 5:
        days = pd.to_datetime(sl["closed_at"]).dt.date.value_counts()
        if not days.empty:
            top_day_count = int(days.iloc[0])
            day_share = top_day_count / len(sl)
            if day_share >= 0.30:
                candidates.append({
                    "cause": f"تركّز يومي ({days.index[0]})",
                    "weight": day_share * 0.55,
                    "evidence": f"{day_share*100:.0f}٪ من الشريحة سُجّلت في يوم واحد — "
                                "يرجّح أن سبب الموجة حدث محدّد بذلك اليوم (تحديث نظام، إعلان، إلخ).",
                })

    if not candidates:
        return []

    # Normalise weights to probabilities, return top 3
    candidates.sort(key=lambda c: -c["weight"])
    candidates = candidates[:3]
    total = sum(c["weight"] for c in candidates) or 1.0
    for c in candidates:
        c["probability"] = round(c["weight"] / total * 100.0, 0)
        del c["weight"]
    return candidates


def rule_based_insights(df: pd.DataFrame) -> list[dict]:
    """Concrete, evidence-backed insights for financial-services CX.

    Each insight carries: title, evidence (with numbers), action (with unit),
    metric (measurable target), kind (icon/color hint), and 2-3 example
    request IDs from the actual data.
    """
    out: list[dict] = []
    if df.empty:
        return out

    k = compute_kpis(df)
    topics = top_recurring_topics(df, top_n=12)

    # 1) Anomalies — surface up to 2
    alerts = detect_weekly_anomalies(df)
    for a in alerts[:2]:
        unit, _ = _topic_action_map(a.value)
        if a.dimension == "category":
            ex = _examples_for(df, category=a.value)
            causes = _attribute_causes(df, category=a.value)
        else:
            ex = _examples_for(df, topic_label=a.value)
            causes = _attribute_causes(df, topic=a.value)
        evidence = (
            f"رصد النظام {a.count} طلب في «{a.value}» مقابل متوسط تاريخي "
            f"{a.baseline_mean:.0f} طلب/أسبوع."
            if a.baseline_mean > 0 else
            f"رصد النظام {a.count} طلب جديد في «{a.value}» — لم يكن مرئياً سابقاً."
        )
        out.append({
            "kind": "anomaly",
            "title": f"تنبيه: ارتفاع في «{a.value}»",
            "evidence": evidence,
            "causes": causes,
            "action": f"تكليف ({unit}) بفحص الموجة فوراً. {a.suggested_action}",
            "metric": "عودة الحجم الأسبوعي إلى المتوسط التاريخي خلال أسبوعين.",
            "examples": ex,
        })

    # 2) Topic with high-severity concentration
    bad = topics[(topics["high_pct"] >= 50) & (topics["count"] >= 30)]
    if not bad.empty:
        r = bad.iloc[0]
        unit, action_body = _topic_action_map(r["topic_label"])
        ex = _examples_for(df, topic_label=r["topic_label"], severity="عالية")
        causes = _attribute_causes(df, topic=r["topic_label"])
        out.append({
            "kind": "risk",
            "title": f"موضوع «{r['topic_label']}» يستنزف موارد الفريق",
            "evidence": (
                f"{int(r['count'])} طلب في الفترة، منها {r['high_pct']:.0f}٪ "
                f"بخطورة عالية — أعلى تركيز خطورة بين الموضوعات."
            ),
            "causes": causes,
            "action": f"إحالة المسار إلى ({unit}). {action_body}",
            "metric": f"خفض نسبة الخطورة العالية في الموضوع إلى ما دون 30٪ خلال 6 أسابيع.",
            "examples": ex,
        })

    # 3) Topic momentum (rising)
    mom = topic_momentum(df, lookback_weeks=4, top_n=5)
    rising = mom[mom["growth_pct"] > 30]
    if not rising.empty:
        r = rising.iloc[0]
        unit, action_body = _topic_action_map(r["topic_label"])
        ex = _examples_for(df, topic_label=r["topic_label"])
        causes = _attribute_causes(df, topic=r["topic_label"])
        out.append({
            "kind": "momentum",
            "title": f"موضوع متصاعد: «{r['topic_label']}»",
            "evidence": (
                f"الطلبات ارتفعت من {int(r['prior'])} إلى {int(r['recent'])} "
                f"(+{r['growth_pct']:.0f}٪) في آخر 4 أسابيع."
            ),
            "causes": causes,
            "action": f"({unit}): {action_body}",
            "metric": "خفض الطلبات في هذا الموضوع بنسبة 30٪ خلال شهر.",
            "examples": ex,
        })

    # 4) Complaint rate + the topic driving it
    if k.pct_complaints >= 15:
        top_complaint_topic = (
            df[df["category"] == "شكوى"]["topic_label"].value_counts().head(1)
        )
        topic_name = top_complaint_topic.index[0] if not top_complaint_topic.empty else "—"
        topic_count = int(top_complaint_topic.iloc[0]) if not top_complaint_topic.empty else 0
        ex = _examples_for(df, category="شكوى", topic_label=topic_name)
        causes = _attribute_causes(df, category="شكوى")
        out.append({
            "kind": "complaints",
            "title": "نسبة الشكاوى تستحق إصلاحاً منهجياً",
            "evidence": (
                f"الشكاوى تمثّل {k.pct_complaints:.0f}٪ من إجمالي الطلبات "
                f"({_count_in(df, 'شكوى')} طلب). أكبر مساهم: «{topic_name}» بـ{topic_count} شكوى."
            ),
            "causes": causes,
            "action": (
                f"تشكيل فريق سريع لمراجعة أسباب الشكاوى المرتبطة بـ«{topic_name}» "
                "وإطلاق إصلاحات قصيرة المدى خلال أسبوعين."
            ),
            "metric": "خفض نسبة الشكاوى الكلية إلى أقل من 12٪ خلال شهرين.",
            "examples": ex,
        })

    # 5) Self-service opportunity
    info_topics = topics[topics["topic_label"].astype(str).str.contains(
        "استعلام|تحديث|الاستفسار", regex=True, na=False
    )]
    info_total = int(info_topics["count"].sum()) if not info_topics.empty else 0
    if info_total >= 200:
        share = info_total / k.total * 100
        topic_name = info_topics["topic_label"].iloc[0]
        ex = _examples_for(df, topic_label=topic_name)
        out.append({
            "kind": "opportunity",
            "title": "فرصة خفض الحمل عبر الخدمة الذاتية",
            "evidence": (
                f"الطلبات الإدارية (استعلامات وتحديث بيانات) تمثّل {share:.0f}٪ "
                f"من الإجمالي ({info_total} طلب) — معظمها قابل للأتمتة."
            ),
            "action": (
                "تفعيل واجهة خدمة ذاتية للاستعلامات المتكرّرة على البوابة وتطبيق "
                "الجوال، مع روابط مباشرة من رسائل الإشعار."
            ),
            "metric": "خفض حجم هذه الطلبات في القناة البشرية بنسبة 25٪ خلال 8 أسابيع.",
            "examples": ex,
        })

    # 6) Recurring beneficiary issues
    recur = find_recurring_cases(df, min_repeats=4, lookback_days=60)
    if not recur.empty:
        top = recur.iloc[0]
        ex = [{"id": int(i), "body": top["phrase"][:70]} for i in (top["sample_ids"] or [])[:3]]
        out.append({
            "kind": "recurring",
            "title": f"حالة متكرّرة: «{top['phrase'][:50]}»",
            "evidence": (
                f"النص نفسه ورد {int(top['count'])} مرة خلال آخر 60 يوماً، "
                f"غالبيته في فئة «{top['top_category']}»، "
                f"وبنسبة خطورة عالية {top['high_pct']:.0f}٪."
            ),
            "action": (
                "فحص ما إذا كان الطلب يأتي من مستفيد واحد متكرّر أم من عدة "
                "مستفيدين بنفس المشكلة — والمعالجة وفقاً لذلك."
            ),
            "metric": "إغلاق جذور المشكلة وإيقاف تكرار النص خلال شهر.",
            "examples": ex,
        })

    return out[:6]


def _count_in(df: pd.DataFrame, category: str) -> int:
    return int((df["category"] == category).sum())


# ---------- Period-over-period comparison ----------

def period_comparison(df: pd.DataFrame, days: int = 14) -> dict:
    """Compare the most recent `days` window with the immediately preceding one."""
    if df.empty:
        return {}
    df = df.copy()
    df["closed_at"] = pd.to_datetime(df["closed_at"])
    end = df["closed_at"].max()
    cur_start = end - pd.Timedelta(days=days)
    prev_end = cur_start
    prev_start = prev_end - pd.Timedelta(days=days)
    cur = df[df["closed_at"] >= cur_start]
    prev = df[(df["closed_at"] >= prev_start) & (df["closed_at"] < prev_end)]

    def block(slice_):
        return {
            "total": int(len(slice_)),
            "complaints": int((slice_["category"] == "شكوى").sum()),
            "high": int((slice_["severity"] == "عالية").sum()),
        }

    cur_b = block(cur); prev_b = block(prev)
    def delta_pct(c, p):
        if p == 0:
            return None if c == 0 else 100.0
        return round((c - p) / p * 100.0, 1)
    return {
        "window_days": days,
        "current_start": cur_start.strftime("%Y-%m-%d"),
        "current_end": end.strftime("%Y-%m-%d"),
        "previous_start": prev_start.strftime("%Y-%m-%d"),
        "previous_end": prev_end.strftime("%Y-%m-%d"),
        "current": cur_b,
        "previous": prev_b,
        "delta_pct": {
            "total": delta_pct(cur_b["total"], prev_b["total"]),
            "complaints": delta_pct(cur_b["complaints"], prev_b["complaints"]),
            "high": delta_pct(cur_b["high"], prev_b["high"]),
        },
    }


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
