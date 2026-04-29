# Changelog

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
