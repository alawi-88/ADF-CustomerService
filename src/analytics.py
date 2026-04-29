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
    pct_insufficient_context: float = 0.0


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
            pct_insufficient_context=0.0,
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
    if "low_content" in df.columns:
        pct_insuff = float(df["low_content"].fillna(False).astype(bool).sum()) / len(df) * 100.0
    else:
        # Fallback: regenerate the heuristic on the fly (no parquet schema upgrade required)
        from src.llm_client import _is_low_content
        pct_insuff = float(df["body"].fillna("").astype(str).map(_is_low_content).sum()) / len(df) * 100.0
    return KPIs(
        total=len(df),
        by_category=by_cat,
        by_severity=by_sev,
        weekly_volume=weekly,
        pct_high_severity=pct_high,
        pct_complaints=pct_compl,
        pct_insufficient_context=pct_insuff,
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
    lang: str = "ar",
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
                        suggested_action=_action_for_anomaly(dim, value, count, mu, sev, lang),
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
                                    (f"ارتفاع متواصل في «{value}» على مدى أسابيع عدة "
                                     "(مستوى أعلى من المتوسط بـ 25٪ أو أكثر). "
                                     if lang != "en" else
                                     f"Sustained elevation in «{value}» over multiple weeks "
                                     "(25% or more above the mean). ")
                                    + _action_for_anomaly(dim, value, count, mu, "متوسطة", lang)
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
                    if lang != "en" else
                    f"A new topic «{topic}» emerged recently with {int(count)} requests — "
                    "investigate the cause before it scales."
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
                    if lang != "en" else
                    f"High-severity hotspot: «{cat}» logs {row['high_pct']:.0f}% "
                    f"of its requests as high severity (vs an average of {avg_high_pct:.0f}% "
                    "across all data). Review the category's root causes and ship fast fixes."
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
                        if lang != "en" else
                        f"Severity is concentrated in «{row['topic_label']}» — "
                        f"{row['high_pct']:.0f}% of its {int(row['count'])} requests "
                        "are high severity. Review the response policy for this topic "
                        "and enable a fast-track channel for it."
                    ),
                ))
        alerts.sort(key=lambda a: -a.count)

    return alerts[:15]


def _action_for_anomaly(dim: str, value: str, count: int,
                         baseline: float, severity: str, lang: str = "ar") -> str:
    delta = count - baseline
    if lang == "en":
        if dim == "category":
            if value == "شكوى":
                return (f"Unusual rise in complaints this week (+{delta:.0f} above the mean). "
                        "Review root causes and launch a proactive outreach campaign.")
            if value == "دعم فني":
                return (f"Tech-support volume is climbing (+{delta:.0f}). "
                        "Audit digital-service availability and channel health.")
            return (f"Unusual rise in category «{value}» (+{delta:.0f}). "
                    "Review the category and run a root-cause analysis.")
        return (f"Unusual rise in topic «{value}» (+{delta:.0f} above the mean). "
                "Recommend a proactive improvement to the related service and a knowledge-base update.")
    # AR (default)
    if dim == "category":
        if value == "شكوى":
            return (f"ارتفاع غير معتاد في الشكاوى هذا الأسبوع (+{delta:.0f} "
                    f"عن المعدل). يُوصى بمراجعة جذور الشكاوى وإطلاق حملة تواصل استباقي.")
        if value == "دعم فني":
            return (f"تصاعد طلبات الدعم الفني (+{delta:.0f}). يُوصى بفحص توافر "
                    f"الخدمات الرقمية وحالة القنوات الإلكترونية.")
        return (f"ارتفاع غير معتاد في فئة «{value}» (+{delta:.0f}). "
                f"يُوصى بمراجعة محتوى الفئة وإجراء تحليل الأسباب الجذرية.")
    return (f"ارتفاع غير معتاد في موضوع «{value}» (+{delta:.0f} عن المعدل). "
            f"يُقترح تحسين استباقي على الخدمة المرتبطة وتحديث قاعدة المعرفة.")


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


# ---------- Semantic linking — group related complaints by meaning, then
#            synthesise what an employee should understand the beneficiary
#            actually wants. Probably the highest-value piece of reasoning
#            in the product: takes raw beneficiary text and gives the
#            ticket-handler a clear intent statement they can act on.

# Each entry: keyword set → (intent_ar, intent_en, suggested_response_ar, suggested_response_en)
_INTENT_PATTERNS: list[tuple] = [
    (("متعسر", "متعثر", "تعسر", "تعثر", "غير قادر", "ما اقدر", "صعوبة في السداد"),
     "المستفيد يواجه صعوبة في الالتزام بجدول السداد ويطلب إعادة جدولة أو إعفاء جزئي.",
     "The beneficiary is unable to keep up with the repayment schedule and is asking for restructuring or partial relief.",
     "اقترح خطة سداد مرنة، اطلب من فريق التحصيل التواصل خلال 48 ساعة، ووثّق وضع المستفيد.",
     "Offer a flexible repayment plan, have collections call within 48 hours, and document the beneficiary's situation."),
    (("ايقاف الحسم", "إيقاف الحسم", "ايقاف الخصم", "إيقاف الخصم", "وقف الخصم", "وقف الحسم"),
     "المستفيد يطلب إيقاف الخصم الشهري على قرضه — إما بسبب ظرف مالي مؤقت أو خلاف على القسط.",
     "The beneficiary wants the monthly loan-instalment deduction halted — either due to a short-term financial issue or a dispute over the instalment amount.",
     "تحقّق من حالة القرض، أوقف الخصم مؤقتاً إن كان نظامياً، وأعد التقييم خلال 7 أيام.",
     "Verify the loan status, pause the deduction temporarily if eligible, and reassess within 7 days."),
    # Generic «خصم» → in our domain almost always means the bank-account
    # deduction taken automatically to cover a loan instalment. We assume
    # that context unless other patterns matched first.
    (("خصم", "حسم"),
     "المستفيد يستفسر/يعترض على عملية خصم تلقائي من حسابه البنكي مرتبطة بقسط القرض. "
     "قد يكون السبب: خصم مكرر، خصم بمبلغ غير صحيح، خصم في توقيت غير متوقع، أو رغبة في إعادة جدولة الأقساط.",
     "The beneficiary is asking about — or disputing — an automatic deduction from their bank account that is tied to a loan instalment. "
     "Likely causes: duplicate deduction, incorrect amount, unexpected timing, or a desire to reschedule the instalments.",
     "افحص سجل الخصم لهذا الحساب لآخر دورتين، قارن المبلغ بالقسط المتعاقد عليه، اتصل بالمستفيد لتأكيد ما يطلبه (مراجعة المبلغ / إيقاف الخصم / إعادة الجدولة) ووجّهه للجهة المناسبة.",
     "Audit the last two deduction cycles on the account, compare the amount against the contracted instalment, call the beneficiary to confirm what they actually want (amount review / pause / rescheduling), and route them to the right unit."),
    (("تأخر", "متأخر", "لم يصل", "لم يُصرف", "بدون رد", "لم يتم", "ما تم"),
     "المستفيد ينتظر إجراءً أو صرفاً ماليّاً ولم يتم في الوقت المتوقع.",
     "The beneficiary is waiting on an action or disbursement that didn't happen on time.",
     "تتبّع الطلب لدى الجهة المختصة، اتصل بالمستفيد خلال 24 ساعة بحالة التحديث، وحدّد سبب التأخير.",
     "Trace the request through the responsible unit, call the beneficiary within 24 hours with a status update, and identify the cause of the delay."),
    (("الغاء عقد", "إلغاء عقد", "الغاء قرض", "إلغاء قرض", "فسخ"),
     "المستفيد يطلب إلغاء/فسخ عقد قائم — قرار مالي حساس يستوجب مراجعة شروط العقد.",
     "The beneficiary is requesting cancellation of an existing contract — a sensitive financial decision that requires reviewing the contract terms.",
     "حوّل إلى الجهة القانونية وفريق الائتمان، راجع شروط الفسخ، وأبلغ المستفيد بالإجراء والمدة المتوقعة.",
     "Refer to legal and credit, review the cancellation terms, and inform the beneficiary of the procedure and expected timeline."),
    (("تحديث رقم", "تحديث جوال", "تحديث بيانات", "تغيير رقم"),
     "المستفيد يطلب تحديث بيانات الاتصال — إجراء روتيني لكنه يلزم لإكمال الخدمات.",
     "The beneficiary wants to update contact details — routine, but required to complete services.",
     "وفّر مسار التحديث الذاتي على البوابة، وتأكّد من تحديث القنوات المرتبطة بعد التحقق.",
     "Offer the self-service update flow on the portal and ensure linked channels reflect the change after verification."),
    (("القروض العادية", "القروض المتخصصة", "نوع القرض", "اي نوع", "ايش الفرق"),
     "المستفيد يستفسر عن نوع/فئة القرض المناسبة لوضعه ولم يحسم اختياره.",
     "The beneficiary is asking which loan product fits their situation and hasn't decided.",
     "أرسل ملخّصاً للأنواع المتاحة وأهليّة كلٍّ منها، واعرض موعداً مع مستشار ائتماني.",
     "Send a summary of available products and eligibility for each, and offer an appointment with a credit advisor."),
    (("اخلاء طرف", "إخلاء طرف"),
     "المستفيد يطلب وثيقة إخلاء طرف بعد إغلاق التزاماته.",
     "The beneficiary is asking for a clearance certificate after closing their obligations.",
     "تحقّق من إغلاق جميع المستحقات، وأصدر الوثيقة إلكترونياً مع روابط التحميل.",
     "Verify all obligations are closed and issue the certificate electronically with download links."),
    (("شكر", "تقدير", "ممتاز"),
     "المستفيد يعبّر عن رضاه — لا حاجة لإجراء تصحيحي.",
     "The beneficiary is expressing satisfaction — no corrective action needed.",
     "أرسل رسالة تقدير قصيرة وأرشف التعليق في سجل التجربة الإيجابية.",
     "Send a short thank-you and archive the comment in the positive-experience log."),
    (("احتيال", "تزوير", "تلاعب", "اخترق"),
     "المستفيد يبلّغ عن احتيال أو تلاعب يحتاج إجراءات أمنية فورية.",
     "The beneficiary is reporting fraud or tampering that requires immediate security action.",
     "صعّد فوراً لإدارة المخاطر، جمّد الحساب وقفل المعاملات حتى التحقق.",
     "Escalate immediately to risk and compliance; freeze the account and any pending transactions until verified."),
]


def _intent_for(text: str) -> tuple[str, str, str, str] | None:
    """Match a (possibly long) request body against the intent patterns."""
    if not text:
        return None
    t = text
    for keys, intent_ar, intent_en, resp_ar, resp_en in _INTENT_PATTERNS:
        for k in keys:
            if k in t:
                return intent_ar, intent_en, resp_ar, resp_en
    return None


def find_related_groups(df: pd.DataFrame,
                        min_size: int = 5,
                        top_n: int = 6,
                        lang: str = "ar",
                        use_llm: bool = True) -> list[dict]:
    """Cluster requests that share an underlying intent and synthesize, in
    the role of the customer-service employee, what the beneficiary likely
    wants.

    Approach: walk through known intent patterns; for each pattern that
    matches at least `min_size` records, build a group with:
      - intent           : "what the beneficiary really wants" (one sentence)
      - employee_response: how an employee should handle it (one sentence)
      - examples         : up to 4 representative IDs+body excerpts
      - count            : how many records share this intent
      - top_category     : dominant category in the group
      - high_pct         : share that are high severity
    """
    if df.empty or "body" not in df.columns:
        return []

    groups: list[dict] = []
    used_ids: set[int] = set()

    for keys, intent_ar, intent_en, resp_ar, resp_en in _INTENT_PATTERNS:
        # Members are records whose body contains any of the keywords AND
        # whose request_id hasn't already been claimed by an earlier intent.
        mask = pd.Series(False, index=df.index)
        for k in keys:
            mask = mask | df["body"].astype(str).str.contains(k, regex=False, na=False)
        members = df[mask]
        members = members[~members["request_id"].astype(int).isin(used_ids)]
        if len(members) < min_size:
            continue
        for rid in members["request_id"].astype(int):
            used_ids.add(int(rid))

        ex_rows = members.sort_values("closed_at", ascending=False).head(4)
        examples = []
        for _, r in ex_rows.iterrows():
            body = str(r.get("body") or "").strip()
            if len(body) > 70:
                body = body[:70].rstrip() + "…"
            examples.append({"id": int(r["request_id"]), "body": body})

        top_cat = members["category"].mode().iloc[0] if not members["category"].mode().empty else None
        high_pct = round((members["severity"] == "عالية").mean() * 100.0, 1)

        groups.append({
            "intent":            intent_en if lang == "en" else intent_ar,
            "employee_response": resp_en   if lang == "en" else resp_ar,
            "match_keywords":    list(keys),
            "count":             int(len(members)),
            "top_category":      top_cat,
            "top_category_en":   None if not top_cat else (
                {"شكوى":"Complaint","استفسار":"Inquiry","اقتراح":"Suggestion",
                 "دعم فني":"Tech support","خدمة مراجع":"Beneficiary service"}.get(top_cat, top_cat)
            ),
            "high_pct":          high_pct,
            "examples":          examples,
        })

    groups.sort(key=lambda g: -g["count"])
    groups = groups[:top_n]

    if use_llm:
        # Layer LLM enrichment for the top groups: pass the actual body
        # samples to a model that already has the domain context embedded
        # in its system prompt. The LLM synthesises the underlying intent
        # the way an experienced service employee would interpret these
        # complaints together — picking up that "خصم" implies bank-account
        # loan-deduction context, even when the body text is one word.
        from src import llm_client  # local import to avoid hard dependency
        for g in groups:
            samples = [e.get("body", "") for e in g.get("examples", [])][:6]
            if not samples:
                continue
            llm_out = llm_client.enrich_group(
                samples,
                g.get("top_category") or "",
                g.get("high_pct") or 0.0,
                language=lang,
            )
            if llm_out and llm_out.get("intent"):
                g["intent"] = llm_out["intent"]
            if llm_out and llm_out.get("employee_response"):
                g["employee_response"] = llm_out["employee_response"]
            if llm_out and llm_out.get("rationale"):
                g["rationale"] = llm_out["rationale"]
            if llm_out and llm_out.get("provider"):
                g["llm_provider"] = llm_out["provider"]

    return groups


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
    alerts = detect_weekly_anomalies(df, lang=lang)
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


_UNIT_AR = {
    "collections": "فريق التحصيل وإدارة الائتمان",
    "risk":        "إدارة المخاطر والامتثال",
    "ops":         "إدارة العمليات المالية",
    "credit":      "إدارة الائتمان",
    "subsidies":   "إدارة برامج الدعم",
    "digital":     "التحول الرقمي",
    "support":     "إدارة التحول الرقمي والدعم الفني",
    "reviewers":   "مكتب خدمة المراجعين",
    "cx":          "إدارة تجربة المستفيد",
    "default":     "الجهة المختصة بالموضوع",
}
_UNIT_EN = {
    "collections": "Collections & credit team",
    "risk":        "Risk & compliance",
    "ops":         "Financial operations",
    "credit":      "Credit team",
    "subsidies":   "Subsidy programs",
    "digital":     "Digital transformation",
    "support":     "Digital channels & tech support",
    "reviewers":   "Beneficiary-service desk",
    "cx":          "Customer-experience team",
    "default":     "Responsible unit for this topic",
}
_ACTION_BODY_AR = {
    "collections": "مراجعة محفظة المتأثرين وتفعيل خطط جدولة سداد مرنة، والتواصل الاستباقي بمستشار مالي مخصّص.",
    "risk":        "تفعيل بروتوكول الكشف عن الاحتيال على المعاملات المرتبطة وإيقاف أي عمليات معلّقة لحين التحقق.",
    "ops":         "مراجعة سجل الخصم لكل طلب، وإجراء التسويات في الحالات الموثّقة خلال 48 ساعة.",
    "credit":      "مراجعة سياسة الاستجابة لطلبات التمويل، وتقصير دورة الاعتماد للحالات الواضحة.",
    "subsidies":   "تحديث مرجعية المعايير على البوابة، وإطلاق دفعة تواصل استباقية للمستفيدين المؤهلين.",
    "digital":     "تفعيل خدمة التحديث الذاتي عبر القناة الرقمية لخفض الطلبات اليدوية.",
    "support":     "مراجعة لوحة توافر الخدمات الرقمية وتحسين رحلة المستخدم في المسارات الأكثر إثارة للطلبات.",
    "reviewers":   "مراجعة قائمة المتطلبات الناقصة الأكثر تكراراً وتبسيط النموذج.",
    "cx":          "تشكيل فريق سريع لتحليل أسباب الشكاوى المتكررة هذا الأسبوع وإطلاق إصلاحات قصيرة المدى.",
    "default":     "إحالة المسار إلى الجهة المعنية مع SLA لا يتجاوز 3 أيام عمل.",
}
_ACTION_BODY_EN = {
    "collections": "Review the affected portfolio, activate flexible repayment plans, and proactively contact each beneficiary with a dedicated financial advisor.",
    "risk":        "Trigger the fraud-detection protocol on related transactions and freeze any pending operations until verified.",
    "ops":         "Audit the deduction record for each request and reconcile documented errors within 48 hours.",
    "credit":      "Revisit the financing-request response policy and shorten the approval cycle for clear-cut cases.",
    "subsidies":   "Refresh the eligibility-criteria reference on the portal and launch a proactive outreach push to qualified beneficiaries.",
    "digital":     "Activate a self-service data-update flow on the digital channel to deflect manual requests.",
    "support":     "Audit the digital-services availability board and improve the user journey on the highest-volume paths.",
    "reviewers":   "Review the most-frequent missing-requirement list and simplify the form.",
    "cx":          "Stand up a quick task force to analyse recurring complaint root causes this week and ship short-term fixes.",
    "default":     "Route the stream to the responsible unit with an SLA of no more than 3 business days.",
}


def _topic_unit_key(topic: str) -> str:
    t = (topic or "")
    if "تعثّر" in t or "تعثر" in t or "السداد" in t or "default" in t.lower() or "repayment" in t.lower(): return "collections"
    if "احتيال" in t or "fraud" in t.lower(): return "risk"
    if "خصم" in t or "deduction" in t.lower(): return "ops"
    if "قرض" in t or "تمويل" in t or "loan" in t.lower() or "financ" in t.lower(): return "credit"
    if "دعم" in t or "subsid" in t.lower() or "support program" in t.lower(): return "subsidies"
    if "تحديث" in t and "بيانات" in t: return "digital"
    if "data update" in t.lower(): return "digital"
    if "قنوات" in t or "الدعم الفني" in t or "tech support" in t.lower() or "channels" in t.lower(): return "support"
    if "خدمة" in t and "مراجع" in t: return "reviewers"
    if "reviewer" in t.lower(): return "reviewers"
    if "شكاو" in t or "complaint" in t.lower(): return "cx"
    return "default"


def _topic_action_map(topic: str, lang: str = "ar") -> tuple[str, str]:
    key = _topic_unit_key(topic)
    if lang == "en":
        return _UNIT_EN[key], _ACTION_BODY_EN[key]
    return _UNIT_AR[key], _ACTION_BODY_AR[key]


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


_CAUSE_TXT = {
    "ar": {
        "dup_cause":  "تكرار النص ذاته «{p}»",
        "dup_ev":     "{n} طلب من أصل {m} يحملون النص نفسه — يرجّح أن المصدر مستفيد متكرر أو مشكلة منهجية واحدة.",
        "topic_cause":"تركّز في موضوع «{t}»",
        "topic_ev":   "{n} طلب ({p}٪ من الشريحة) ينتمون لموضوع واحد — يرجّح أن المصدر مشكلة محدّدة في هذا المسار.",
        "cat_cause":  "كل الطلبات تقريباً في فئة «{c}»",
        "cat_ev":     "{p}٪ من الشريحة من فئة واحدة — التحدي تشغيلي في فريق هذه الفئة وليس عاماً.",
        "kw_cause":   "إشارات حساسة في النص (مثل «{k}»)",
        "kw_ev":      "كلمة «{k}» وردت {n} مرة في الشريحة — تشير إلى مشكلة مالية/تشغيلية تستدعي التصعيد.",
        "day_cause":  "تركّز يومي ({d})",
        "day_ev":     "{p}٪ من الشريحة سُجّلت في يوم واحد — يرجّح أن سبب الموجة حدث محدّد بذلك اليوم.",
    },
    "en": {
        "dup_cause":  "The same text is repeated «{p}»",
        "dup_ev":     "{n} of {m} requests carry identical text — likely a single repeat beneficiary or a systemic issue.",
        "topic_cause":"Concentration in topic «{t}»",
        "topic_ev":   "{n} requests ({p}% of the slice) belong to a single topic — likely a specific issue in that flow.",
        "cat_cause":  "Almost all requests are in category «{c}»",
        "cat_ev":     "{p}% of the slice come from one category — operational challenge for that team, not a general issue.",
        "kw_cause":   "Risk-laden language in the body (e.g. «{k}»)",
        "kw_ev":      "The word «{k}» appears {n} times in the slice — points to a financial/operational issue worth escalating.",
        "day_cause":  "Same-day concentration ({d})",
        "day_ev":     "{p}% of the slice were logged on a single day — likely a specific event that day (release, announcement, outage).",
    },
}


def _attribute_causes(df: pd.DataFrame, *, category: str | None = None,
                      topic: str | None = None, lang: str = "ar") -> list[dict]:
    """Decompose 'why is this X concentrated/elevated?' into ranked causes."""
    if df.empty:
        return []
    C = _CAUSE_TXT.get(lang, _CAUSE_TXT["ar"])

    sl = df
    if category: sl = sl[sl["category"] == category]
    if topic:    sl = sl[sl["topic_label"] == topic]
    if sl.empty:
        return []

    candidates: list[dict] = []

    if "body" in sl.columns:
        norm = sl["body"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
        top_phrase = norm.value_counts().head(1)
        if not top_phrase.empty:
            phrase, n = top_phrase.index[0], int(top_phrase.iloc[0])
            share = n / len(sl)
            if n >= 3 and share >= 0.05:
                candidates.append({
                    "cause":    C["dup_cause"].format(p=phrase[:60]),
                    "weight":   share * 0.9,
                    "evidence": C["dup_ev"].format(n=n, m=len(sl)),
                })

    if topic is None and "topic_label" in sl.columns:
        top_topic = sl["topic_label"].value_counts().head(1)
        if not top_topic.empty:
            tname, tn = top_topic.index[0], int(top_topic.iloc[0])
            tshare = tn / len(sl)
            if tshare >= 0.20:
                candidates.append({
                    "cause":    C["topic_cause"].format(t=tname),
                    "weight":   tshare * 0.85,
                    "evidence": C["topic_ev"].format(n=tn, p=int(tshare*100)),
                })

    if category is None and "category" in sl.columns:
        top_cat = sl["category"].value_counts().head(1)
        if not top_cat.empty:
            cn = int(top_cat.iloc[0])
            cshare = cn / len(sl)
            if cshare >= 0.50:
                candidates.append({
                    "cause":    C["cat_cause"].format(c=top_cat.index[0]),
                    "weight":   cshare * 0.7,
                    "evidence": C["cat_ev"].format(p=int(cshare*100)),
                })

    sev_kw = ["متعسر", "تعثر", "متأخر", "تأخر", "إيقاف", "ايقاف", "خصم", "احتيال",
              "رفض", "استرداد", "مستحقات"]
    body_text = sl["body"].astype(str).str.cat(sep=" ")
    hits = [(kw, body_text.count(kw)) for kw in sev_kw if body_text.count(kw) >= 3]
    if hits:
        hits.sort(key=lambda x: -x[1])
        top_kw, top_c = hits[0]
        share = top_c / len(sl)
        candidates.append({
            "cause":    C["kw_cause"].format(k=top_kw),
            "weight":   min(0.6, share * 0.6),
            "evidence": C["kw_ev"].format(k=top_kw, n=top_c),
        })

    if "closed_at" in sl.columns and len(sl) >= 5:
        days = pd.to_datetime(sl["closed_at"]).dt.date.value_counts()
        if not days.empty:
            top_day_count = int(days.iloc[0])
            day_share = top_day_count / len(sl)
            if day_share >= 0.30:
                candidates.append({
                    "cause":    C["day_cause"].format(d=days.index[0]),
                    "weight":   day_share * 0.55,
                    "evidence": C["day_ev"].format(p=int(day_share*100)),
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


_TXT = {
    "ar": {
        "anomaly_title":     "تنبيه: ارتفاع في «{v}»",
        "anomaly_ev":        "رصد النظام {n} طلب في «{v}» مقابل متوسط تاريخي {b} طلب/أسبوع.",
        "anomaly_ev_new":    "رصد النظام {n} طلب جديد في «{v}» — لم يكن مرئياً سابقاً.",
        "anomaly_action":    "تكليف ({u}) بفحص الموجة فوراً. {sa}",
        "anomaly_metric":    "عودة الحجم الأسبوعي إلى المتوسط التاريخي خلال أسبوعين.",
        "risk_title":        "موضوع «{t}» يستنزف موارد الفريق",
        "risk_ev":           "{n} طلب في الفترة، منها {p}٪ بخطورة عالية — أعلى تركيز خطورة بين الموضوعات.",
        "risk_action":       "إحالة المسار إلى ({u}). {ab}",
        "risk_metric":       "خفض نسبة الخطورة العالية في الموضوع إلى ما دون 30٪ خلال 6 أسابيع.",
        "mom_title":         "موضوع متصاعد: «{t}»",
        "mom_ev":            "الطلبات ارتفعت من {p} إلى {r} (+{g}٪) في آخر 4 أسابيع.",
        "mom_action":        "({u}): {ab}",
        "mom_metric":        "خفض الطلبات في هذا الموضوع بنسبة 30٪ خلال شهر.",
        "compl_title":       "نسبة الشكاوى تستحق إصلاحاً منهجياً",
        "compl_ev":          "الشكاوى تمثّل {p}٪ من إجمالي الطلبات ({n} طلب). أكبر مساهم: «{t}» بـ{tn} شكوى.",
        "compl_action":      "تشكيل فريق سريع لمراجعة أسباب الشكاوى المرتبطة بـ«{t}» وإطلاق إصلاحات قصيرة المدى خلال أسبوعين.",
        "compl_metric":      "خفض نسبة الشكاوى الكلية إلى أقل من 12٪ خلال شهرين.",
        "opp_title":         "فرصة خفض الحمل عبر الخدمة الذاتية",
        "opp_ev":            "الطلبات الإدارية (استعلامات وتحديث بيانات) تمثّل {p}٪ من الإجمالي ({n} طلب) — معظمها قابل للأتمتة.",
        "opp_action":        "تفعيل واجهة خدمة ذاتية للاستعلامات المتكرّرة على البوابة وتطبيق الجوال، مع روابط مباشرة من رسائل الإشعار.",
        "opp_metric":        "خفض حجم هذه الطلبات في القناة البشرية بنسبة 25٪ خلال 8 أسابيع.",
        "rec_title":         "حالة متكرّرة: «{p}»",
        "rec_ev":            "النص نفسه ورد {n} مرة خلال آخر 60 يوماً، غالبيته في فئة «{c}»، وبنسبة خطورة عالية {h}٪.",
        "rec_action":        "فحص ما إذا كان الطلب يأتي من مستفيد واحد متكرّر أم من عدة مستفيدين بنفس المشكلة — والمعالجة وفقاً لذلك.",
        "rec_metric":        "إغلاق جذور المشكلة وإيقاف تكرار النص خلال شهر.",
    },
    "en": {
        "anomaly_title":     "Alert: rise in «{v}»",
        "anomaly_ev":        "{n} requests in «{v}» vs a historical average of {b}/week.",
        "anomaly_ev_new":    "{n} new requests in «{v}» — not previously visible.",
        "anomaly_action":    "Task ({u}) to investigate the surge immediately. {sa}",
        "anomaly_metric":    "Return weekly volume to the historical average within two weeks.",
        "risk_title":        "Topic «{t}» is draining team resources",
        "risk_ev":           "{n} requests in the period; {p}% are high-severity — the highest severity concentration among topics.",
        "risk_action":       "Route this stream to ({u}). {ab}",
        "risk_metric":       "Drop the topic's high-severity rate below 30% within 6 weeks.",
        "mom_title":         "Rising topic: «{t}»",
        "mom_ev":            "Requests rose from {p} to {r} (+{g}%) over the last 4 weeks.",
        "mom_action":        "({u}): {ab}",
        "mom_metric":        "Reduce requests in this topic by 30% within a month.",
        "compl_title":       "Complaint share warrants a systemic fix",
        "compl_ev":          "Complaints are {p}% of all requests ({n} total). Biggest driver: «{t}» with {tn} complaints.",
        "compl_action":      "Stand up a quick task force to review root causes of complaints tied to «{t}» and ship short-term fixes within two weeks.",
        "compl_metric":      "Reduce overall complaint share below 12% within two months.",
        "opp_title":         "Opportunity: deflect routine load to self-service",
        "opp_ev":            "Administrative requests (inquiries and data updates) are {p}% of total ({n} requests) — most are automatable.",
        "opp_action":        "Roll out a self-service surface for the most-repeated inquiries on the portal and mobile app, with deep links from notification messages.",
        "opp_metric":        "Reduce these requests on the human channel by 25% within 8 weeks.",
        "rec_title":         "Recurring case: «{p}»",
        "rec_ev":            "The same text appeared {n} times in the last 60 days, mostly in category «{c}», with {h}% high severity.",
        "rec_action":        "Investigate whether the requests come from one repeat beneficiary or multiple beneficiaries hitting the same issue — handle accordingly.",
        "rec_metric":        "Close the underlying root cause and stop the text from recurring within a month.",
    },
}


def dashboard_ai_summary(df: pd.DataFrame, lang: str = "ar") -> dict:
    """Narrative summary of the current slice — what the data says, in
    a few sentences, plus 3-5 highlight bullets the user should know."""
    if df.empty:
        msg = ("لا توجد بيانات ضمن النطاق الحالي." if lang != "en"
               else "No data in the current scope.")
        return {"summary": msg, "highlights": []}

    from src.llm_client import CATEGORY_EN, SEVERITY_EN

    k = compute_kpis(df)
    by_cat = k.by_category
    top_cat_ar = by_cat.index[0] if not by_cat.empty else "—"
    top_cat = (CATEGORY_EN.get(top_cat_ar, top_cat_ar) if lang == "en" else top_cat_ar)
    top_share = (by_cat.iloc[0] / k.total * 100) if k.total else 0
    n_cats = int(by_cat.size)

    topics = top_recurring_topics(df, top_n=2)
    top_topic_ar = topics.iloc[0]["topic_label"] if not topics.empty else "—"
    top_topic = (_localize_value(df, top_topic_ar, "topic_label", lang)
                 if not topics.empty else top_topic_ar)
    top_topic_count = int(topics.iloc[0]["count"]) if not topics.empty else 0

    fc = forecast_weekly(df, horizon=1)
    next_week = int(round(fc["forecast"]["y"][0])) if fc["forecast"]["y"] else None

    recur = find_recurring_cases(df, min_repeats=4, lookback_days=60)
    top_recur = recur.iloc[0] if not recur.empty else None

    mom = topic_momentum(df, lookback_weeks=4, top_n=3)
    rising_topic = None
    rising_growth = None
    if not mom.empty:
        r0 = mom.iloc[0]
        if r0["growth_pct"] > 25:
            rising_topic = _localize_value(df, r0["topic_label"], "topic_label", lang)
            rising_growth = int(r0["growth_pct"])

    if lang == "en":
        sentences = [
            f"You handled {k.total:,} requests across {n_cats} categories.",
            f"«{top_cat}» dominates at {top_share:.0f}% of volume.",
            f"{k.pct_high_severity:.0f}% are flagged high-severity, "
            f"concentrated in «{top_topic}» ({top_topic_count} requests).",
        ]
        if next_week is not None:
            sentences.append(f"Forecast for next week: ~{next_week} requests.")
    else:
        sentences = [
            f"تمّت معالجة {k.total:,} طلب في الفترة موزّعة على {n_cats} فئة.",
            f"الفئة المسيطرة «{top_cat}» بنسبة {top_share:.0f}٪ من الحجم.",
            f"{k.pct_high_severity:.0f}٪ منها بخطورة عالية، "
            f"يتركّز معظمها في موضوع «{top_topic}» ({top_topic_count} طلب).",
        ]
        if next_week is not None:
            sentences.append(f"التنبؤ للأسبوع القادم: نحو {next_week} طلب.")

    summary = " ".join(sentences)

    highlights: list[dict] = []
    if rising_topic:
        highlights.append({
            "kind": "rising",
            "text": (f"«{rising_topic}» rising +{rising_growth}% in the last 4 weeks"
                     if lang == "en" else
                     f"«{rising_topic}» في تصاعد +{rising_growth}٪ خلال آخر ٤ أسابيع"),
        })
    if top_recur is not None and int(top_recur["count"]) >= 5:
        ph = str(top_recur["phrase"])[:50]
        highlights.append({
            "kind": "recurring",
            "text": (f"«{ph}» repeated {int(top_recur['count'])} times — likely repeat issue"
                     if lang == "en" else
                     f"«{ph}» متكرر {int(top_recur['count'])} مرة — على الأرجح مشكلة منهجية"),
        })
    if k.pct_high_severity >= 25:
        highlights.append({
            "kind": "risk",
            "text": (f"{k.pct_high_severity:.0f}% of total flagged high — review escalation policy"
                     if lang == "en" else
                     f"{k.pct_high_severity:.0f}٪ من الإجمالي مصنّفة عالية الخطورة — راجع سياسة التصعيد"),
        })
    if next_week is not None:
        highlights.append({
            "kind": "forecast",
            "text": (f"Plan staffing for ~{next_week} requests next week"
                     if lang == "en" else
                     f"خطّط الطاقم لاستقبال نحو {next_week} طلب الأسبوع القادم"),
        })

    return {"summary": summary, "highlights": highlights[:4]}


def ticket_ai_view(record: dict, df: pd.DataFrame, lang: str = "ar") -> dict:
    """Per-ticket AI summary + suggestions + recommendation.
    `record` is the row dict; `df` is the full enriched dataset for
    context (similar-tickets count, etc.)."""
    from src.llm_client import CATEGORY_EN, SEVERITY_EN

    body = (record.get("body") or "").strip()
    cat_ar = record.get("category") or ""
    cat = (CATEGORY_EN.get(cat_ar, cat_ar) if lang == "en" else cat_ar)
    sev_ar = record.get("severity") or ""
    sev = (SEVERITY_EN.get(sev_ar, sev_ar) if lang == "en" else sev_ar)
    topic_ar = (record.get("topic_label_ar") or record.get("topic_label") or "")
    topic = (record.get("topic_label_en") if lang == "en" else topic_ar) or topic_ar
    reason = (record.get(f"severity_reason_{lang}")
              or record.get("severity_reason_ar") or "")
    action = (record.get(f"recommended_action_{lang}")
              or record.get("recommended_action_ar") or "")

    # Similar-ticket counts and sample IDs
    similar_count = 0
    similar_ids: list[int] = []
    if topic_ar and "topic_label" in df.columns:
        sim = df[(df["topic_label"] == topic_ar)
                 & (df["request_id"] != record.get("request_id"))]
        similar_count = int(len(sim))
        similar_ids = sim["request_id"].head(3).astype(int).tolist()

    # Domain-specific suggestions based on topic
    suggestions = _ticket_suggestions(topic_ar, cat_ar, body, lang)

    # Compose the summary paragraph
    excerpt = body if len(body) <= 70 else body[:70].rstrip() + "…"
    if lang == "en":
        sim_str = (f"This is one of {similar_count} similar requests about «{topic}»."
                   if similar_count else "No closely-similar requests found in the dataset.")
        summary = (
            f"This is a {cat.lower()} request from the beneficiary about «{excerpt}». "
            f"Severity: {sev} — {reason} {sim_str}"
        )
    else:
        sim_str = (f"يوجد {similar_count} طلب مشابه في موضوع «{topic}»."
                   if similar_count else "لا توجد طلبات مشابهة قريبة في البيانات.")
        summary = (
            f"طلب من نوع «{cat}» يخصّ «{excerpt}». "
            f"الخطورة: {sev} — {reason} {sim_str}"
        )

    return {
        "summary": summary.strip(),
        "suggestions": suggestions,
        "recommendation": action,
        "similar_count": similar_count,
        "similar_ids": similar_ids,
    }


def _ticket_suggestions(topic_ar: str, cat_ar: str, body: str, lang: str) -> list[str]:
    """Return up to 4 domain-specific suggestions for a ticket. Mostly
    'questions to ask the beneficiary' or 'data points to verify'."""
    t = (topic_ar or "")
    items: list[tuple[str, str]] = []  # (ar, en)

    def _add(ar: str, en: str) -> None:
        items.append((ar, en))

    # Repayment / loan deduction
    if "تعثّر" in t or "تعثر" in t or "السداد" in t:
        _add("اطلب من المستفيد توضيح سبب التعثّر (ظرف مؤقت / دائم).",
             "Ask the beneficiary to explain the cause of the default (temporary / persistent).")
        _add("راجع تاريخ الأقساط المتأخرة في النظام قبل الرد.",
             "Check the late-instalments history in the system before replying.")
        _add("اعرض خطة جدولة مرنة وسجّل الموافقة على المسار.",
             "Offer a flexible repayment plan and log the chosen path.")
    elif "خصم" in t:
        _add("تحقّق من سجل الخصم لآخر دورتين على الحساب.",
             "Verify the deduction log for the last two cycles on the account.")
        _add("قارن المبلغ المخصوم مع القسط المتعاقد عليه.",
             "Compare the deducted amount against the contracted instalment.")
        _add("اسأل المستفيد: هل ما يطلبه مراجعة المبلغ، إيقاف الخصم، أم إعادة جدولة؟",
             "Ask the beneficiary: are they requesting an amount review, a deduction halt, or rescheduling?")
    elif "احتيال" in t:
        _add("صعّد فوراً إلى إدارة المخاطر والامتثال.",
             "Escalate to risk and compliance immediately.")
        _add("جمّد المعاملات المعلّقة حتى التحقق.",
             "Freeze pending transactions until verification.")
        _add("اطلب من المستفيد قائمة العمليات المشبوهة بالتاريخ والمبلغ.",
             "Ask the beneficiary for a list of suspect transactions with dates and amounts.")
    elif "قرض" in t or "تمويل" in t or cat_ar == "خدمة مراجع":
        _add("اطلب نوع التمويل المرغوب وقيمة المشروع التقديرية.",
             "Ask for the desired financing product and the estimated project value.")
        _add("تحقّق من اكتمال ملف المستفيد قبل الإحالة.",
             "Verify the beneficiary's file is complete before routing.")
        _add("شارك جدول الأهلية للمنتجات الممكنة.",
             "Share the eligibility table for available products.")
    elif "تأخر" in t or "delay" in t.lower():
        _add("تتبّع الطلب لدى الجهة المختصة وحدّد سبب التأخير.",
             "Trace the request through the responsible unit and identify the cause of the delay.")
        _add("اتصل بالمستفيد بنتيجة المراجعة قبل نهاية اليوم التالي.",
             "Call the beneficiary with the review outcome before end of next business day.")
    elif "تحديث" in t and "بيانات" in t:
        _add("وجّه المستفيد إلى مسار التحديث الذاتي على البوابة.",
             "Direct the beneficiary to the self-service update flow on the portal.")
        _add("تحقّق من مزامنة القنوات بعد التحديث.",
             "Verify cross-channel sync after the update.")
    elif "شكاو" in t or cat_ar == "شكوى":
        _add("اطلب من المستفيد تفاصيل المشكلة والتاريخ والمبلغ إن وُجد.",
             "Ask the beneficiary for details, date, and amount (if any).")
        _add("اتصل استباقياً قبل تصعيد التذكرة.",
             "Place a proactive call before escalating the ticket.")
        _add("سجّل الشكوى في نظام الجودة لأغراض التحليل.",
             "Log the complaint in the QA system for analytics purposes.")
    else:
        _add("اطلب من المستفيد تفاصيل أوضح حول طلبه.",
             "Ask the beneficiary for clearer details about their request.")
        _add("ابحث عن تذاكر مشابهة سابقة قبل الرد.",
             "Search for similar prior tickets before replying.")

    return [(en if lang == "en" else ar) for ar, en in items[:4]]


def _localize_value(df: pd.DataFrame, value: str, dimension: str, lang: str) -> str:
    """Translate a category or topic value when rendering in English."""
    if lang != "en":
        return value
    from src.llm_client import CATEGORY_EN  # local import to avoid cycle at module top
    if dimension == "category":
        return CATEGORY_EN.get(value, value)
    if dimension == "topic_label":
        if "topic_label_ar" in df.columns and "topic_label_en" in df.columns:
            m = df[df["topic_label_ar"] == value]
            if not m.empty:
                en = m["topic_label_en"].iloc[0]
                if en:
                    return en
    return value


def rule_based_insights(df: pd.DataFrame, lang: str = "ar") -> list[dict]:
    """Concrete, evidence-backed insights — produced fully in `lang`.

    Each insight carries: title, evidence (with numbers), causes (80/20),
    action (with unit), metric (measurable target), kind (icon/color hint),
    and example request IDs from the actual data.
    """
    L = _TXT.get(lang, _TXT["ar"])
    out: list[dict] = []
    if df.empty:
        return out

    k = compute_kpis(df)
    topics = top_recurring_topics(df, top_n=12)

    # 1) Anomalies — surface up to 2
    alerts = detect_weekly_anomalies(df, lang=lang)
    for a in alerts[:2]:
        unit, _ = _topic_action_map(a.value, lang)
        if a.dimension == "category":
            ex = _examples_for(df, category=a.value)
            causes = _attribute_causes(df, category=a.value, lang=lang)
        else:
            ex = _examples_for(df, topic_label=a.value)
            causes = _attribute_causes(df, topic=a.value, lang=lang)
        ev_template = L["anomaly_ev"] if a.baseline_mean > 0 else L["anomaly_ev_new"]
        v_local = _localize_value(df, a.value, a.dimension, lang)
        out.append({
            "kind": "anomaly",
            "title":   L["anomaly_title"].format(v=v_local),
            "evidence": ev_template.format(n=a.count, v=v_local, b=int(a.baseline_mean)),
            "causes":   causes,
            "action":   L["anomaly_action"].format(u=unit, sa=a.suggested_action),
            "metric":   L["anomaly_metric"],
            "examples": ex,
        })

    # 2) Topic with high-severity concentration
    bad = topics[(topics["high_pct"] >= 50) & (topics["count"] >= 30)]
    if not bad.empty:
        r = bad.iloc[0]
        unit, action_body = _topic_action_map(r["topic_label"], lang)
        ex = _examples_for(df, topic_label=r["topic_label"], severity="عالية")
        causes = _attribute_causes(df, topic=r["topic_label"], lang=lang)
        t_local = _localize_value(df, r["topic_label"], "topic_label", lang)
        out.append({
            "kind": "risk",
            "title":    L["risk_title"].format(t=t_local),
            "evidence": L["risk_ev"].format(n=int(r["count"]), p=int(r["high_pct"])),
            "causes":   causes,
            "action":   L["risk_action"].format(u=unit, ab=action_body),
            "metric":   L["risk_metric"],
            "examples": ex,
        })

    # 3) Topic momentum (rising)
    mom = topic_momentum(df, lookback_weeks=4, top_n=5)
    rising = mom[mom["growth_pct"] > 30]
    if not rising.empty:
        r = rising.iloc[0]
        unit, action_body = _topic_action_map(r["topic_label"], lang)
        ex = _examples_for(df, topic_label=r["topic_label"])
        causes = _attribute_causes(df, topic=r["topic_label"], lang=lang)
        t_local = _localize_value(df, r["topic_label"], "topic_label", lang)
        out.append({
            "kind": "momentum",
            "title":    L["mom_title"].format(t=t_local),
            "evidence": L["mom_ev"].format(p=int(r["prior"]), r=int(r["recent"]), g=int(r["growth_pct"])),
            "causes":   causes,
            "action":   L["mom_action"].format(u=unit, ab=action_body),
            "metric":   L["mom_metric"],
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
        causes = _attribute_causes(df, category="شكوى", lang=lang)
        topic_local = _localize_value(df, topic_name, "topic_label", lang)
        out.append({
            "kind": "complaints",
            "title":    L["compl_title"],
            "evidence": L["compl_ev"].format(p=int(k.pct_complaints),
                                              n=_count_in(df, "شكوى"),
                                              t=topic_local, tn=topic_count),
            "causes":   causes,
            "action":   L["compl_action"].format(t=topic_local),
            "metric":   L["compl_metric"],
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
            "title":    L["opp_title"],
            "evidence": L["opp_ev"].format(p=int(share), n=info_total),
            "action":   L["opp_action"],
            "metric":   L["opp_metric"],
            "examples": ex,
        })

    # 6) Recurring beneficiary issues
    recur = find_recurring_cases(df, min_repeats=4, lookback_days=60)
    if not recur.empty:
        top = recur.iloc[0]
        ex = [{"id": int(i), "body": top["phrase"][:70]} for i in (top["sample_ids"] or [])[:3]]
        out.append({
            "kind": "recurring",
            "title":    L["rec_title"].format(p=top["phrase"][:50]),
            "evidence": L["rec_ev"].format(n=int(top["count"]),
                                            c=top["top_category"],
                                            h=int(top["high_pct"])),
            "action":   L["rec_action"],
            "metric":   L["rec_metric"],
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
