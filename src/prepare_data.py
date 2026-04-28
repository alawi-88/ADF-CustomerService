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


def _load_one(path: Path) -> pd.DataFrame:
    log.info("loading %s", path.name)
    df = pd.read_excel(path)
    if df.shape[1] < 5:
        raise ValueError(f"{path.name}: expected at least 5 columns, got {df.shape[1]}")
    body_col = df.columns[3]
    df = df.rename(columns={**COLUMN_MAP, body_col: "body"})
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
    return df


def load_all_raw(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    files = sorted([p for p in raw_dir.glob("*.xlsx") if not p.name.startswith("~$")])
    if not files:
        raise FileNotFoundError(f"no .xlsx files found in {raw_dir}")
    frames = [_load_one(p) for p in files]
    df = pd.concat(frames, ignore_index=True)
    before = len(df)
    df = df.dropna(subset=["closed_at", "request_id"]).reset_index(drop=True)
    df = df.sort_values("closed_at").drop_duplicates("request_id", keep="last")
    log.info("merged %d row(s) from %d file(s); kept %d unique after dedup",
             before, len(files), len(df))
    return df.reset_index(drop=True)


def _enrich_one(row, prefer_llm: bool) -> pd.Series:
    e = llm_client.enrich_record(row.category, row.body, prefer_llm=prefer_llm)
    return pd.Series({
        "severity": e.severity,
        "severity_reason": e.severity_reason,
        "topic_label": e.topic_label,
        "recommended_action": e.recommended_action,
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
        # Only reuse rows that have ALL required enrichment fields. If the
        # schema evolved (e.g., we added severity_reason), force re-enrichment.
        required = ("severity", "severity_reason", "topic_label", "recommended_action", "ai_source")
        if all(c in prev.columns for c in required):
            for _, r in prev.iterrows():
                if pd.isna(r["severity"]) or pd.isna(r["severity_reason"]):
                    continue
                cached[int(r["request_id"])] = {
                    "severity": r["severity"],
                    "severity_reason": r["severity_reason"],
                    "topic_label": r["topic_label"],
                    "recommended_action": r["recommended_action"],
                    "ai_source": r["ai_source"],
                }

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
