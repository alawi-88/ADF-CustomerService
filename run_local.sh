#!/usr/bin/env bash
# Local launcher for the customer-experience analytics dashboard.
#   ./run_local.sh prepare    # one-time data enrichment
#   ./run_local.sh            # launch FastAPI dashboard

set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"

if [ ! -d ".venv" ]; then
  echo "[setup] creating virtualenv..."
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo "[setup] installing requirements..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

case "${1:-run}" in
  prepare)
    echo "[prepare] enriching raw Excel into Parquet cache..."
    python -m src.prepare_data
    ;;
  run|"")
    if [ ! -f "data/processed/enriched.parquet" ]; then
      echo "[prepare] cache missing — running prepare_data first..."
      python -m src.prepare_data
    fi
    echo "[run] launching dashboard on http://localhost:8501"
    export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
    uvicorn src.app:app --host 0.0.0.0 --port 8501 --no-access-log
    ;;
  *)
    echo "Usage: ./run_local.sh [prepare|run]"
    exit 1
    ;;
esac
