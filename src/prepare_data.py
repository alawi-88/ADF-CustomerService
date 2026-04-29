"""Data enrichment: any *.xlsx in data/raw/  →  enriched Parquet cache.

Loads every Excel file in data/raw/, merges them, deduplicates by
request_id, then enriches with severity / topic / recommended_action.
This makes the dashboard incremental: drop new exports into data/raw/
(or upload them through the UI) and re-run prepare to refresh.

Usage:
    python -m src.prepare_data            # merge all xlsx in data/raw/
    python -m src.prepare_data --no-llm   # force rule-based
    python -m src.prepare_data --only NEW.xlsx  # only enrich new rows; keep cached enrichments for existing IDs
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

from src import analytics, llm_client

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("prepare")

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
DEFAULT_OUT = PROCESSED_DIR / "enriched.parquet"

# Source columns (Arabic) -> normalized English names
COLUMN_MAP = {
    "الرقم": "request_id",
    "الطلب": "request_type",
    "العنوان": "category",
    "تاريخ الانتهاء": "closed_at",
    # Body column (col D) is positionally picked up because the source
    # export sometimes has no header for that column.
}


# Tolerant column matching — accept reasonable variations of the official names
_HEADER_ALIASES = {
    "request_id":   ["الرقم", "id", "ticket id", "ticket_id", "request id", "request_id", "رقم الطلب", "رقم الطلب رقم", "#"],
    "request_type": ["الطلب", "type", "request type", "request_type", "نوع الطلب", "نوع"],
    "category":     ["العنوان", "category", "subject", "title", "الفئة", "نوع الفئة", "العنوان الفرعي"],
    "body":         ["body", "نص", "نص الطلب", "تفاصيل", "details", "description", "موضوع", "ملاحظات", "محتوى"],
    "closed_at":    ["تاريخ الانتهاء", "closed at", "closed_at", "date", "تاريخ", "تاريخ الإغلاق", "ended at"],
}


def _resolve_columns(df: pd.DataFrame, fname: str) -> dict:
    """Map raw column names to the canonical 5 we need.
    Falls back to positional when header isn't recognised."""
    cols = list(df.columns)
    norm = [str(c).strip().lower() for c in cols]
    resolved: dict[str, str] = {}
    for canonical, aliases in _HEADER_ALIASES.items():
        for a in aliases:
            a_n = a.strip().lower()
            for orig, n in zip(cols, norm):
                if n == a_n:
                    resolved[canonical] = orig
                    break
            if canonical in resolved:
                break
    # If body still missing, fall back to positional col index 3 (the
    # original export sometimes ships the body column unnamed).
    if "body" not in resolved and len(cols) >= 4:
        resolved["body"] = cols[3]
    missing = [c for c in ("request_id", "request_type", "category", "body", "closed_at") if c not in resolved]
    if missing:
        raise ValueError(
            f"{fname}: cannot find required column(s): {missing}. "
            f"Found headers: {list(cols)}. "
            f"Expected (with aliases): {list(_HEADER_ALIASES.keys())}."
        )
    return resolved


def _load_one(path: Path) -> pd.DataFrame:
    log.info("loading %s", path.name)
    df = pd.read_excel(path)
    if df.shape[1] < 5:
        raise ValueError(f"{path.name}: expected at least 5 columns, got {df.shape[1]}")
    resolved = _resolve_columns(df, path.name)
    rename = {orig: canonical for canonical, orig in resolved.items()}
    df = df.rename(columns=rename)
    keep = ["request_id", "request_type", "category", "body", "closed_at"]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name}: missing required columns: {missing}")
    df = df[keep].copy()
    df["category"] = df["category"].astype(str).str.strip()
    df["body"] = df["body"].astype(str).str.strip()
    df["closed_at"] = pd.to_datetime(df["closed_at"], errors="coerce")
    df["request_id"] = pd.to_numeric(df["request_id"], errors="coerce").astype("Int64")
    df["_source_file"] = path.name
    # Track missing dates rather than silently dropping them. We still need a
    # usable date for time-series rollups, so we impute with the max date in
    # the file but flag the imputation so the UI can surface the count.
    df["_date_missing"] = df["closed_at"].isna()
    if df["_date_missing"].any():
        max_date = df["closed_at"].max()
        if pd.isna(max_date):
            max_date = pd.Timestamp.now().normalize()
        df.loc[df["_date_missing"], "closed_at"] = max_date
    return df


def load_all_raw(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    files = sorted([p for p in raw_dir.glob("*.xlsx") if not p.name.startswith("~$")])
    if not files:
        raise FileNotFoundError(f"no .xlsx files found in {raw_dir}")
    frames = [_load_one(p) for p in files]
    df = pd.concat(frames, ignore_index=True)
    before = len(df)
    # Only drop rows missing the unique ID — those cannot be uniquely tracked.
    # Rows missing dates are kept (with imputed closed_at + _date_missing flag).
    df = df.dropna(subset=["request_id"]).reset_index(drop=True)
    df = df.sort_values("closed_at").drop_duplicates("request_id", keep="last")
    n_imputed = int(df.get("_date_missing", pd.Series(False, index=df.index)).sum())
    log.info("merged %d row(s) from %d file(s); %d unique kept; %d had missing dates (imputed).",
             before, len(files), len(df), n_imputed)
    return df.reset_index(drop=True)


def _enrich_one(row, prefer_llm: bool) -> pd.Series:
    e = llm_client.enrich_record(row.category, row.body, prefer_llm=prefer_llm)
    low = bool(llm_client._is_low_content(row.body))
    return pd.Series({
        "severity": e.severity,
        "subcategory": getattr(e, "subcategory", "unspecified"),
        "low_content": low,
        # AR + EN parallel fields
        "severity_reason_ar":   e.severity_reason_ar,
        "severity_reason_en":   e.severity_reason_en,
        "topic_label_ar":       e.topic_label_ar,
        "topic_label_en":       e.topic_label_en,
        "recommended_action_ar":e.recommended_action_ar,
        "recommended_action_en":e.recommended_action_en,
        # legacy names kept = AR (back-compat for any code/UX that hasn't switched yet)
        "severity_reason":   e.severity_reason_ar,
        "topic_label":       e.topic_label_ar,
        "recommended_action":e.recommended_action_ar,
        "ai_source": e.source,
    })


def run(prefer_llm: Optional[bool] = None,
        out_path: Path = DEFAULT_OUT,
        limit: Optional[int] = None) -> dict:
    """Programmatic entry point used by the upload endpoint."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = load_all_raw()
    if limit:
        df = df.head(limit)

    status = llm_client.runtime_status()
    use_llm = bool(status["ollama_available"]) if prefer_llm is None else prefer_llm
    log.info("enriching %d records (use_llm=%s, ollama=%s)",
             len(df), use_llm, status["ollama_available"])

    # Reuse cached enrichments where the same request_id already exists in
    # the previous parquet, to avoid re-running the LLM for unchanged rows.
    cached: dict[int, dict] = {}
    if out_path.exists():
        prev = pd.read_parquet(out_path)
        required = ("severity", "severity_reason_ar", "severity_reason_en",
                    "topic_label_ar", "topic_label_en",
                    "recommended_action_ar", "recommended_action_en", "ai_source")
        if all(c in prev.columns for c in required):
            for _, r in prev.iterrows():
                if pd.isna(r["severity"]) or pd.isna(r["severity_reason_ar"]):
                    continue
                cached[int(r["request_id"])] = {col: r[col] for col in required}
                # legacy aliases
                cached[int(r["request_id"])]["severity_reason"]    = r["severity_reason_ar"]
                cached[int(r["request_id"])]["topic_label"]        = r["topic_label_ar"]
                cached[int(r["request_id"])]["recommended_action"] = r["recommended_action_ar"]

    new_rows = []
    reused = 0
    for _, row in df.iterrows():
        rid = int(row["request_id"])
        if rid in cached:
            new_rows.append(cached[rid])
            reused += 1
        else:
            new_rows.append(_enrich_one(row, use_llm).to_dict())
    enriched = pd.DataFrame(new_rows)
    df = pd.concat([df.reset_index(drop=True), enriched.reset_index(drop=True)], axis=1)
    log.info("reused enrichments for %d cached rows; computed %d new",
             reused, len(df) - reused)

    log.info("running TF-IDF + KMeans clustering ...")
    df = analytics.assign_topic_clusters(df, text_col="body", category_col="category", k=8)

    df["week_start"] = (
        pd.to_datetime(df["closed_at"]).dt.to_period("W-SAT").dt.start_time
    )

    log.info("writing %s", out_path)
    df.to_parquet(out_path, index=False)

    summary = {
        "rows": int(len(df)),
        "by_category": {k: int(v) for k, v in df["category"].value_counts().items()},
        "by_severity": {k: int(v) for k, v in df["severity"].value_counts().items()},
        "files": sorted({p.name for p in RAW_DIR.glob("*.xlsx") if not p.name.startswith("~$")}),
        "reused_enrichments": reused,
    }
    log.info("done — %s", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Enrich raw Excel data into Parquet.")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--no-llm", action="store_true", help="Force rule-based enrichment.")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)
    run(prefer_llm=False if args.no_llm else None,
        out_path=args.out, limit=args.limit)
    return 0


if __name__ == "__main__":
    sys.exit(main())
