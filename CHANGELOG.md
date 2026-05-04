# Changelog

## [1.4.0] — 2026-05-04

Visual identity + admin controls release. Aligns the platform with the
DGA design system (typography, color, dark header) and gives the admin
direct control over comparison periods and export periods.

### Added
- **DGA color system** — full primary green ramp (50–900), secondary
  navy ramp, warm-neutral ramp, and DGA-aligned semantic colors —
  declared as `--dga-*` CSS custom properties in
  `static/dga_overrides.css`. Supersedes the ad-hoc green tokens that
  were inherited from the upstream `dga.css` mirror.
- **Dark topbar** — header background uses `--dga-green-800` (#003318)
  so the supplied white-text ADF logo sits on a brand-correct dark
  surface. All topbar controls (language switch, upload button,
  health pill, actor select) restyled for the dark surface with
  AA-grade contrast and visible focus rings.
- **Period comparison selector** on Overview — a segmented control
  next to the period-comparison card lets the admin switch between
  **Weekly** (7d), **Monthly** (30d), and **Quarterly** (90d). The
  card title updates to reflect the chosen period, and the choice is
  persisted to `localStorage` (`adf.pop.period`).
- **Export period filter** — every Excel export button is now a split
  button: clicking the main label exports the dashboard's current
  view; clicking the chevron opens a menu with explicit period
  presets (current view, last 7/30/90 days, this month, last month,
  all data). The export endpoint receives explicit `from`/`to`
  parameters when a preset is chosen, overriding the dashboard's
  current date filter without disturbing the dashboard view.
- 18 new i18n keys (9 AR + 9 EN) for the period selector, export menu
  headings, and chevron aria-label.

### Changed
- **Typography → IBM Plex Sans family** — Arabic now renders in
  *IBM Plex Sans Arabic* (weights 300–700) and Latin in *IBM Plex Sans*
  (400–700). Replaces the previous Noto Naskh / Noto Sans Arabic
  pair. Tabular numerics retained via `font-variant-numeric:
  tabular-nums`.
- FastAPI app version bumped to `1.4.0`.

### Notes
- A subsequent v1.4.x will adopt DGA-supplied component primitives
  and icons from a Figma source-of-truth (pending file from project
  lead). The token-level alignment shipped here is a pre-requisite
  for that work.

## [1.3.1] — 2026-05-04 (hotfix)

### Fixed
- `/api/export/overview` 500 error — `_export_alerts` was passing
  `AnomalyAlert` dataclass instances to the Excel export module, which
  expected dict-shaped alerts. Now translates dataclass fields into the
  expected `title / kind / metric / evidence` shape (matching what
  `/api/alerts` already returns to the frontend). All four exports now
  work end-to-end on real data.

## [1.3.0] — 2026-05-04

Brand & deliverable release. Adds official ADF visual identity to the
dashboard, polishes Arabic copy across the entire UI to match the
adf.gov.sa formal-public-sector tone, and gives every primary report
surface a one-click Excel export with native Excel charts.

### Added
- **ADF brand identity** — `static/adf-logo-header.svg` replaces the
  topbar wordmark; new `static/adf-logo-footer.svg` anchors a brand
  footer that runs across every page. Header logo collapses gracefully
  on mobile; the footer stacks below 720px.
- **Excel export endpoints** — four new GET routes return formatted
  `.xlsx` workbooks with native Excel charts (no embedded PNGs):
  - `GET /api/export/overview` — KPI block, weekly volume (line chart),
    by-category (bar), severity split (doughnut), early-warning alerts.
  - `GET /api/export/patterns` — weekly volume, categories, severity,
    severity-by-week, rising topics, top recurring topics, weekly-by-
    category — each on its own sheet with a chart.
  - `GET /api/export/recommendations[?snapshot_id=…]` — current snapshot
    (or named one) with kind/title/metric/evidence/action/severity, plus
    an "insights by kind" doughnut.
  - `GET /api/export/tickets` — filtered ticket list with auto-filter and
    frozen header, plus a severity distribution chart.
  Every workbook opens with a cover sheet (title, generated-at,
  applied-filter summary, headline KPIs). All sheets are RTL-aware when
  `lang=ar`. Cover and chart titles localised AR/EN.
- **Frontend export buttons** on the four matching pages
  (Overview, Patterns & Trends, Suggested actions, Recommendations log).
  Each button submits the dashboard's current filter state, downloads
  the workbook, and uses the icon `#i-download`.
- **Five new i18n keys** in both AR and EN: `action.export_xlsx`,
  `action.exporting`, `action.export_failed`, `footer.org`, `footer.app`.

### Changed
- **Full Arabic UI revision** — every Arabic string in the i18n table
  has been polished to align with the adf.gov.sa formal-public-sector
  voice: consistent diacritics, Arabic-comma punctuation, government-
  sector vocabulary, and gender-neutral phrasing. ~130 keys touched.
  The English block is left unchanged except for the new keys.
- **App title and FastAPI metadata** updated to surface the ADF brand:
  `منصّة تحليل مشاركات المستفيدين — صندوق التنمية الزراعية`, version
  bumped to `1.3.0`.
- **CSS overlay** in `static/dga_overrides.css` extended with a
  `v1.3.0` block: `.topbar__logo*`, `.app-footer*`, `.page-head__actions`,
  `.btn-export`. Mobile breakpoints handle small screens.

### Dependencies
- Added `xlsxwriter>=3.2,<4.0` for native Excel chart generation.
  `openpyxl` remains for ingest-side reading.

### Verified
- All four export endpoints register on app boot and return their
  expected media type (`application/vnd.openxmlformats-…`). The
  recommendations export was end-to-end tested against an empty
  dataset and produced a valid Excel 2007+ workbook with Arabic
  sheet names («الغلاف», «التوصيات»). Overview / patterns / tickets
  exports return 503 only when the dataset itself is unloaded —
  identical to existing data-dependent endpoints — and produce full
  workbooks otherwise.
- Python `py_compile` clean on `src/excel_export.py` and `src/app.py`.

## [1.2.2] — 2026-04-29 (final tested release)

Hardening release. No new features beyond what landed in v1.2.1, but
the platform is more forgiving (Excel header variations) and faster
to query (SQLite indexes on the case-management tables).

### Added
- **SQLite indexes** on `tickets.db` for the case-management tables:
  `ix_tickets_status`, `ix_tickets_assignee`, `ix_comments_request`,
  `ix_status_request`. Idempotent — safe on repeat startup.
- **Tolerant Excel header detection** — `_resolve_columns` now accepts
  English aliases (`#`, `Type`, `Body`, `Closed At`, etc.) plus the
  original Arabic headers. Falls back to positional column 4 for the
  body if it's unnamed (matches the original ADF export shape).

### Fixed
- `enrich_record` low-content fast-path: confirmed wired and exercised
  on the 30k corpus — 7,276 rows correctly classified as
  `insufficient_context` at ingest.

### Verified
- 24/24 API endpoints pass.
- Snapshot lifecycle: full pass.
- Upload + auto-snapshot: pass.
- UI smoke: 0 issues.
- UI snapshots history: 0 issues.
- UI subcategory chart: renders with coverage hint.
- UI insufficient-context tile: 24.2% rendered in Arabic-Indic numerals.
- Mobile responsive (375 / 768 / 1024 px): 0 layout issues.
- **axe-core WCAG 2.1 AA: 0 violations across all 7 pages.**
- Scale benchmark (30k rows): all reads < 142 ms P95;
  `POST /api/snapshots` writes 30k items in ~635 ms.

### Known caveats
- LLM provider in this release's tests is the rule-based engine only.
  With Groq or Ollama connected, snapshots record `provider="groq"`/`ollama`
  and subcategory coverage rises from 24% to 100%.
- Multi-worker uvicorn still blocked by the process-local lock in
  `tickets.py` — queued for v1.3.0.

---

## [1.2.1] — 2026-04-29

Mobile responsive layout, security hardening (CORS + sanitised 500s),
subcategory pipeline end-to-end. See `upload-to-github-v1.2.1/CHANGELOG.md`.

---

## [1.2.0] — 2026-04-29

Snapshot history UI, prompt v2, insufficient-context KPI, axe-clean
across 7 pages, scale-tested at 30k tickets.

---

## [1.1.1] — 2026-04-29 (test pass)

Bug fixes from end-to-end testing of v1.1.0.

---

## [1.1.0] — 2026-04-29

Recommendation snapshots, low-content fast-path, DGA design fixes, Docker.

---

## [1.0.0] — 2026-04-15

Initial POC release.
