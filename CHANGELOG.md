# Changelog

## [1.7.0] — 2026-05-04

### Fixed
- **Emoji glyphs replaced with monochrome SVG icons.** Four inline
  emoji that bypassed the icon system and rendered as colored Unicode
  pictographs on every OS — `⚠ ✎ 💬` — have been swapped to proper
  outline-stroke `<svg><use/></svg>` references that obey the v1.6.0
  icon style spec (1.5 stroke, currentColor, no fills).
  - Data-quality pill `⚠` → `#i-alert`
  - Severity-override marker `✎` → `#i-edit` (new symbol)
  - Ticket-comment count `💬` → `#i-comment` (new symbol)
- New `.icon--xs` size modifier (13×13) for inline-with-text icon
  usage, so replacement glyphs sit on the baseline like the original
  emoji did, but stay monochrome.

## [1.6.0] — 2026-05-04

Icon style unification — DGA-aligned outline-only spec. Per project
direction the canonical Figma assets aren't being adopted on this
release; instead every icon is normalised to a single visual rule so
later swap-in of the DGA SVGs is a trivial path replace.

### Changed
- **Single icon spec across the platform**: 24×24 grid, `fill="none"`,
  `stroke="currentColor"`, `stroke-width="1.5"`, round line caps and
  joins. Applies to all 15 icon symbols (`#i-overview` through
  `#i-chevron-down`).
- **No tinted icons anywhere**. The upstream `dga.css` rule
  `.nav__item.is-active .nav__icon { color: var(--primary); }` tinted
  the active sidebar icon green. v1.6.0 overrides that and any
  similar rule via a defensive `.icon, .icon *, .nav__icon { fill:
  none !important; stroke: currentColor; … }` block in
  `dga_overrides.css`. Active state is now signalled via text weight
  and background only — icons stay monochrome.
- Active sidebar item gains `font-weight: 600` to compensate for the
  removed icon-tint affordance.

### Notes
- This release is icon-only. The full DGA component primitives (from
  the Components Library Figma file) are still pending: the project
  isn't using the Figma Dev Mode MCP server, so programmatic asset
  extraction isn't available, and a manual SVG export was deferred.

## [1.5.0] — 2026-05-04

Readability and copy-quality release. Addresses the on-screen issues
caught during dashboard review.

### Fixed
- **Empty drawer fields no longer render as "—"**. The records
  side-panel previously showed `الموضوع: —`, `سبب الخطورة: —`, and
  `الإجراء: —` when those fields were absent (which happens when no
  LLM provider is configured). v1.5.0 omits the line entirely when
  the underlying value is missing or just a placeholder dash.

### Changed
- **Western (Hindu-Arabic) numerals everywhere**. `1, 2, 3 …` instead
  of `١, ٢, ٣ …` in both Arabic and English modes. The legacy
  `toAr()` is now a no-op so all numeric tokens, IDs, percentages,
  KPI values, dates, and chart axes render with `0–9`.
- **No diacritics (تشكيل)**. Stripped 27 fatha/damma/kasra/shadda/
  sukun/tanween marks from the entire template. The Arabic copy is
  now plain script as requested.
- **Removed the word "رؤى"** across the UI. Replaced with `تحليلات`
  (or `الملاحظات` / `النتائج` depending on context).
- **Shorter, more everyday wording** in the Arabic UI. Examples:
  `محرّك التحليل → حالة التحليل`, `استعلام مباشر → اسأل بياناتك`,
  `الإجراءات الأعلى أثرًا → الإجراءات الأكثر تأثيرًا`,
  `موضوعات متصاعدة → موضوعات في ارتفاع`,
  `سرد مكثّف → سرد قصير`.
- **AI summary renders as bullet points**, not a paragraph. The
  summary text is split on sentence boundaries (`.`, `!`, `?`,
  `؟`) and each sentence becomes a list item with a green DGA
  bullet — scannable at a glance.

### Notes
- Front-end-only release; no backend route changes. The Excel export
  module, the period comparison endpoint, and the four export routes
  are unchanged from v1.4.0.

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
