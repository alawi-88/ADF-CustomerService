# Customer-Experience Analytics Dashboard

Bilingual (Arabic / English) analytics dashboard for customer-service
submissions — complaints, inquiries, suggestions. Ingests Excel exports,
classifies severity with full justification, surfaces actionable
recommendations, forecasts next-week volume, and detects recurring cases.

Designed to run entirely locally with optional cloud-LLM acceleration.

---

## Highlights

- **Multi-provider LLM chain** — Groq cloud first (configurable model
  cascade), local Ollama as fallback, deterministic rule engine as a
  safety net so the dashboard works even with no AI configured.
- **Severity with justification** — every classification includes a
  short reason that quotes the actual request body.
- **Actionable insights** — synthesizes signals from the full corpus into
  3–5 concrete recommendations (evidence + action + measurable metric).
- **Drill-down everywhere** — click any KPI, chart slice, alert, or
  recurring-case row to open a side drawer with the underlying records.
- **Forecasting** — exponential-smoothing forecast with a confidence band
  for the next 1–2 weeks, plus a per-category outlook.
- **Recurring-case intelligence** — clusters of near-identical request
  bodies recurring within the lookback window — likely repeat
  beneficiaries or systemic issues.
- **Bilingual UI** — full Arabic ↔ English toggle: text direction,
  alignment, digit system (Arabic-Indic ↔ Western), date formatting,
  chart axes, and LLM response language all switch together.
- **Excel ingestion** — drop new `.xlsx` files in via the upload UI;
  records are merged with existing data and deduplicated by ID.

---

## Stack

Python 3.9+ · FastAPI + Uvicorn · pandas · scikit-learn · Plotly · Jinja2 ·
optional Ollama (local LLM) · optional Groq (cloud LLM).

---

## Project layout

```
.
├── README.md
├── requirements.txt
├── run_local.sh
├── data/
│   ├── raw/             # drop .xlsx exports here (gitignored)
│   └── processed/       # parquet cache (gitignored, regenerated)
├── src/
│   ├── llm_client.py    # provider chain + rule engine
│   ├── prepare_data.py  # ingest, dedupe, enrich
│   ├── analytics.py     # KPIs, clustering, anomalies, forecast
│   └── app.py           # FastAPI server
├── static/
│   └── dga.css          # design-system styles
└── templates/
    └── index.html       # single-page dashboard
```

---

## Required Excel format

Every uploaded file must have these five columns, in this order:

| # | Column (AR)         | Meaning              |
|---|---------------------|----------------------|
| 1 | `الرقم`              | unique request ID    |
| 2 | `الطلب`              | request type         |
| 3 | `العنوان`            | category             |
| 4 | (body — any header) | short body / topic   |
| 5 | `تاريخ الانتهاء`     | closed timestamp     |

Recognised categories: `شكوى`, `استفسار`, `اقتراح`, `دعم فني`, `خدمة مراجع`.

---

## Quick start

```bash
git clone <this-repo>
cd <this-repo>
chmod +x run_local.sh

# 1. Drop one or more .xlsx files into data/raw/
mkdir -p data/raw && cp /path/to/your_export.xlsx data/raw/

# 2. (Optional) Configure Groq for fast cloud inference
export GROQ_API_KEY=gsk_...
export GROQ_MODELS=llama-3.3-70b-versatile,llama-3.1-8b-instant,gemma2-9b-it

# 3. Launch
./run_local.sh prepare    # one-time enrichment
./run_local.sh            # serves on http://localhost:8501
```

The dashboard uses `GROQ_API_KEY` first if set. If absent or rate-limited,
it falls back to a local Ollama daemon (`OLLAMA_BASE_URL`, `OLLAMA_MODEL`).
If neither is reachable, the rule-based engine still produces classified
output for every record.

### Optional: install local Ollama

```bash
# macOS
brew install --cask ollama-app
ollama serve &
ollama pull qwen2.5:7b-instruct
```

---

## Pages

| Page                  | What it shows |
|-----------------------|---------------|
| Overview              | KPIs, AI insights, weekly trend, category mix, severity mix, two-week forecast with confidence band. |
| Patterns & trends     | Top recurring topics, severity over time, rising-topics momentum, category trends, severity matrix, recurring-case clusters. |
| Early warning         | Anomaly alerts (z-score on weekly counts) with suggested actions. |
| Suggested actions     | Per-record table: ID, body, category, topic, severity, severity reason, recommended action. |
| Free query            | Natural-language question → answered using current filtered slice. |
| Data management       | List of source files, upload, reprocess. |

---

## API endpoints

```
GET  /api/health              liveness + active LLM provider
GET  /api/meta                date range, categories, severities
GET  /api/kpis                headline metrics
GET  /api/categories          counts per category
GET  /api/severity            counts per severity
GET  /api/weekly              weekly volume (overall)
GET  /api/weekly_by_cat       weekly volume per category
GET  /api/severity_weekly     weekly volume per severity (stacked)
GET  /api/topics              top recurring topics
GET  /api/topic_momentum      rising topics (last vs prior period)
GET  /api/category_matrix     category × severity heatmap
GET  /api/alerts              anomaly alerts
GET  /api/insights            actionable AI insights
GET  /api/forecast            week-ahead volume forecast
GET  /api/recurring_cases     recurring-text clusters
GET  /api/records             paginated records (drill-down)
POST /api/qa                  free-form question (accepts language)
POST /api/upload              add new .xlsx
POST /api/refresh             re-enrich without uploading
GET  /api/files               list ingested files
```

All GET endpoints accept the same filter set: `from`, `to`, `category[]`,
`severity[]`. Records also accept `topic[]`, `body_contains`, `week_start`,
`only_high`.

---

## Notes

- All processing is local except the optional Groq path. With Groq disabled
  no data leaves the machine.
- Customer-data files in `data/raw/` are gitignored — they should never be
  committed.
- The rule engine is honest about its limits: when no LLM is reachable,
  every reasoning string is generated deterministically and clearly
  references the body text it acted on.
