"""ADF Customer-Service POC — Excel export module (v1.3.0).

Builds richly-formatted .xlsx workbooks for each of the four user-facing
report surfaces, with native Excel charts that render whether the file is
opened in Microsoft Excel, Numbers, LibreOffice, or Google Sheets.

Public entry points:

    build_overview_workbook(df, kpis, weekly, categories, severity, alerts) -> bytes
    build_patterns_workbook(df, weekly, categories, severity, severity_weekly,
                            momentum, topics, weekly_by_cat) -> bytes
    build_recommendations_workbook(snapshot, insights, kpis) -> bytes
    build_tickets_workbook(df, filters_summary) -> bytes

All functions return raw .xlsx bytes; the caller wraps with a StreamingResponse.

Design notes
------------
- Native Excel charts only — no embedded PNGs. Means file size is small
  (~10–40 KB) and charts stay editable.
- ADF green primary (#006C35) used for accents and chart series.
- RTL-aware: workbook right_to_left flag set when `lang == "ar"`.
- Header row formatted as bold white on ADF green; rows banded.
- Date columns formatted as YYYY-MM-DD; numbers grouped with thousand sep.
- KPI tiles rendered as a small grid with large value cells.
- Cover sheet on every workbook with title, generated-at timestamp,
  filter summary, and an at-a-glance KPI block.
"""

from __future__ import annotations

import io
from datetime import datetime
from typing import Any, Iterable, Mapping, Optional

import pandas as pd
import xlsxwriter

# ----- Brand tokens (kept in sync with static/dga_overrides.css) -----------
ADF_GREEN = "#006C35"
ADF_GREEN_DARK = "#005628"
ADF_GREEN_LIGHT = "#E5F2EB"
ADF_TEXT = "#1A1A1A"
ADF_MUTED = "#5C6770"
ADF_HIGH = "#AF0818"
ADF_MED = "#B07800"
ADF_LOW = "#006604"
ADF_BORDER = "#D6D9DC"


# =============================================================================
# Workbook scaffolding
# =============================================================================

def _new_workbook(lang: str = "ar") -> tuple[xlsxwriter.Workbook, io.BytesIO]:
    """Create an in-memory workbook with RTL flag matching `lang`."""
    buf = io.BytesIO()
    wb = xlsxwriter.Workbook(buf, {"in_memory": True, "default_date_format": "yyyy-mm-dd"})
    return wb, buf


def _formats(wb: xlsxwriter.Workbook) -> dict[str, Any]:
    """Reusable cell formats."""
    return {
        "title": wb.add_format({
            "bold": True, "font_size": 18, "font_color": ADF_GREEN_DARK,
            "align": "right", "valign": "vcenter",
        }),
        "subtitle": wb.add_format({
            "italic": True, "font_size": 10, "font_color": ADF_MUTED,
            "align": "right", "valign": "vcenter",
        }),
        "section": wb.add_format({
            "bold": True, "font_size": 12, "font_color": "white",
            "bg_color": ADF_GREEN, "align": "right", "valign": "vcenter",
            "border": 1, "border_color": ADF_GREEN_DARK,
        }),
        "header": wb.add_format({
            "bold": True, "font_color": "white", "bg_color": ADF_GREEN,
            "align": "center", "valign": "vcenter",
            "border": 1, "border_color": ADF_GREEN_DARK,
            "text_wrap": True,
        }),
        "cell": wb.add_format({
            "align": "right", "valign": "vcenter",
            "border": 1, "border_color": ADF_BORDER,
            "text_wrap": True,
        }),
        "cell_alt": wb.add_format({
            "align": "right", "valign": "vcenter",
            "bg_color": ADF_GREEN_LIGHT,
            "border": 1, "border_color": ADF_BORDER,
            "text_wrap": True,
        }),
        "num": wb.add_format({
            "align": "center", "valign": "vcenter",
            "num_format": "#,##0", "border": 1, "border_color": ADF_BORDER,
        }),
        "pct": wb.add_format({
            "align": "center", "valign": "vcenter",
            "num_format": "0.0%", "border": 1, "border_color": ADF_BORDER,
        }),
        "date": wb.add_format({
            "align": "center", "valign": "vcenter",
            "num_format": "yyyy-mm-dd",
            "border": 1, "border_color": ADF_BORDER,
        }),
        "kpi_label": wb.add_format({
            "bold": True, "font_size": 11, "font_color": ADF_MUTED,
            "align": "right", "valign": "vcenter",
        }),
        "kpi_value": wb.add_format({
            "bold": True, "font_size": 22, "font_color": ADF_GREEN_DARK,
            "align": "right", "valign": "vcenter",
        }),
        "sev_high": wb.add_format({
            "bold": True, "font_color": "white", "bg_color": ADF_HIGH,
            "align": "center", "valign": "vcenter",
            "border": 1, "border_color": ADF_BORDER,
        }),
        "sev_med": wb.add_format({
            "bold": True, "font_color": "white", "bg_color": ADF_MED,
            "align": "center", "valign": "vcenter",
            "border": 1, "border_color": ADF_BORDER,
        }),
        "sev_low": wb.add_format({
            "bold": True, "font_color": "white", "bg_color": ADF_LOW,
            "align": "center", "valign": "vcenter",
            "border": 1, "border_color": ADF_BORDER,
        }),
        "footer": wb.add_format({
            "italic": True, "font_size": 9, "font_color": ADF_MUTED,
            "align": "right", "valign": "vcenter",
        }),
    }


def _add_cover_sheet(
    wb: xlsxwriter.Workbook,
    fmts: dict,
    *,
    title: str,
    subtitle: str,
    filter_summary: Optional[Mapping[str, Any]] = None,
    kpis: Optional[Mapping[str, Any]] = None,
    lang: str = "ar",
) -> None:
    ws = wb.add_worksheet("الغلاف" if lang == "ar" else "Cover")
    ws.right_to_left() if lang == "ar" else None
    ws.hide_gridlines(2)
    ws.set_column("A:A", 4)
    ws.set_column("B:B", 28)
    ws.set_column("C:C", 28)
    ws.set_column("D:D", 28)
    ws.set_column("E:E", 4)

    ws.set_row(1, 30); ws.set_row(2, 22)
    ws.merge_range("B2:D2", title, fmts["title"])
    ws.merge_range("B3:D3", subtitle, fmts["subtitle"])

    # Generated-at line
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    label = "تم التوليد في" if lang == "ar" else "Generated at"
    ws.merge_range("B5:D5", f"{label}: {now_str}", fmts["footer"])

    row = 7
    if filter_summary:
        ws.merge_range(row, 1, row, 3,
                       "ملخّص عوامل التصفية" if lang == "ar" else "Applied filters",
                       fmts["section"])
        row += 1
        for k, v in filter_summary.items():
            ws.write(row, 1, k, fmts["kpi_label"])
            ws.merge_range(row, 2, row, 3, str(v), fmts["cell"])
            row += 1
        row += 1

    if kpis:
        ws.merge_range(row, 1, row, 3,
                       "أبرز المؤشرات" if lang == "ar" else "Headline KPIs",
                       fmts["section"])
        row += 1
        for k, v in kpis.items():
            ws.set_row(row, 32)
            ws.write(row, 1, k, fmts["kpi_label"])
            try:
                num = float(v)
                if num != num:
                    raise ValueError
                ws.merge_range(row, 2, row, 3, num, fmts["kpi_value"])
            except (TypeError, ValueError):
                ws.merge_range(row, 2, row, 3, str(v) if v is not None else "—",
                               fmts["kpi_value"])
            row += 1


def _write_table(
    ws,
    fmts: dict,
    headers: list[str],
    rows: Iterable[Iterable[Any]],
    *,
    start_row: int = 0,
    start_col: int = 0,
    col_widths: Optional[list[int]] = None,
    col_formats: Optional[list[Optional[str]]] = None,
) -> int:
    """Write a banded data table; returns the row index AFTER the last data row."""
    for j, h in enumerate(headers):
        ws.write(start_row, start_col + j, h, fmts["header"])
    if col_widths:
        for j, w in enumerate(col_widths):
            ws.set_column(start_col + j, start_col + j, w)

    r = start_row + 1
    for ri, row in enumerate(rows):
        cell_fmt = fmts["cell"] if ri % 2 == 0 else fmts["cell_alt"]
        for j, v in enumerate(row):
            kind = (col_formats[j] if col_formats and j < len(col_formats) else None)
            if kind == "num" and v is not None and v != "":
                ws.write_number(r, start_col + j, float(v), fmts["num"])
            elif kind == "pct" and v is not None and v != "":
                ws.write_number(r, start_col + j, float(v), fmts["pct"])
            elif kind == "date" and v is not None and v != "":
                try:
                    if isinstance(v, str):
                        d = datetime.strptime(v[:10], "%Y-%m-%d").date()
                    else:
                        d = pd.Timestamp(v).date()
                    ws.write_datetime(r, start_col + j, datetime(d.year, d.month, d.day),
                                      fmts["date"])
                except Exception:
                    ws.write(r, start_col + j, str(v), fmts["cell"])
            elif kind == "sev":
                f = {"high": fmts["sev_high"], "med": fmts["sev_med"],
                     "low": fmts["sev_low"]}.get(str(v).lower(), cell_fmt)
                ws.write(r, start_col + j, v, f)
            else:
                ws.write(r, start_col + j, "" if v is None else v, cell_fmt)
        r += 1
    return r


# =============================================================================
# Public builders — one per page surface
# =============================================================================

def build_overview_workbook(
    *,
    df: Optional[pd.DataFrame],
    kpis: Mapping[str, Any],
    weekly: list[dict],
    categories: list[dict],
    severity: list[dict],
    alerts: Optional[list[dict]] = None,
    filter_summary: Optional[Mapping[str, Any]] = None,
    lang: str = "ar",
) -> bytes:
    wb, buf = _new_workbook(lang)
    fmts = _formats(wb)

    L = _labels(lang)

    # Cover
    headline = {
        L["kpi.total"]: kpis.get("total", 0),
        L["kpi.complaints"]: f"{kpis.get('complaints_pct', 0):.1f}%",
        L["kpi.high"]: kpis.get("high_severity", 0),
        L["kpi.cats"]: kpis.get("active_categories", 0),
        L["kpi.insufficient"]: kpis.get("insufficient_pct", 0),
    }
    _add_cover_sheet(
        wb, fmts,
        title=L["overview.title"],
        subtitle=L["overview.subtitle"],
        filter_summary=filter_summary,
        kpis=headline,
        lang=lang,
    )

    # Weekly volume
    ws_w = wb.add_worksheet(L["sheet.weekly"])
    ws_w.right_to_left() if lang == "ar" else None
    ws_w.hide_gridlines(2)
    ws_w.merge_range("A1:D1", L["chart.weekly"], fmts["section"])
    rows = [[r.get("week") or r.get("date"), r.get("count", 0)] for r in (weekly or [])]
    end = _write_table(ws_w, fmts, [L["col.week"], L["col.count"]], rows,
                       start_row=2, col_widths=[18, 14], col_formats=["date", "num"])
    if rows:
        chart = wb.add_chart({"type": "line"})
        chart.add_series({
            "name": L["chart.weekly"],
            "categories": [ws_w.name, 3, 0, end - 1, 0],
            "values":     [ws_w.name, 3, 1, end - 1, 1],
            "line":       {"color": ADF_GREEN, "width": 2.25},
            "marker":     {"type": "circle", "size": 6,
                           "border": {"color": ADF_GREEN_DARK},
                           "fill":   {"color": ADF_GREEN}},
        })
        chart.set_title({"name": L["chart.weekly"]})
        chart.set_x_axis({"name": L["col.week"]})
        chart.set_y_axis({"name": L["col.count"]})
        chart.set_legend({"none": True})
        ws_w.insert_chart(2, 4, chart, {"x_scale": 1.4, "y_scale": 1.4})

    # Categories
    ws_c = wb.add_worksheet(L["sheet.categories"])
    ws_c.right_to_left() if lang == "ar" else None
    ws_c.hide_gridlines(2)
    ws_c.merge_range("A1:D1", L["chart.categories"], fmts["section"])
    rows = [[r.get("name"), r.get("count", 0)] for r in (categories or [])]
    end = _write_table(ws_c, fmts, [L["col.category"], L["col.count"]], rows,
                       start_row=2, col_widths=[28, 14], col_formats=[None, "num"])
    if rows:
        chart = wb.add_chart({"type": "bar"})
        chart.add_series({
            "name": L["chart.categories"],
            "categories": [ws_c.name, 3, 0, end - 1, 0],
            "values":     [ws_c.name, 3, 1, end - 1, 1],
            "fill":       {"color": ADF_GREEN},
            "border":     {"color": ADF_GREEN_DARK},
        })
        chart.set_title({"name": L["chart.categories"]})
        chart.set_legend({"none": True})
        ws_c.insert_chart(2, 4, chart, {"x_scale": 1.4, "y_scale": 1.4})

    # Severity
    ws_s = wb.add_worksheet(L["sheet.severity"])
    ws_s.right_to_left() if lang == "ar" else None
    ws_s.hide_gridlines(2)
    ws_s.merge_range("A1:D1", L["chart.severity"], fmts["section"])
    rows = [[r.get("severity"), r.get("count", 0)] for r in (severity or [])]
    end = _write_table(ws_s, fmts, [L["col.severity"], L["col.count"]], rows,
                       start_row=2, col_widths=[14, 14], col_formats=["sev", "num"])
    if rows:
        chart = wb.add_chart({"type": "doughnut"})
        chart.add_series({
            "name": L["chart.severity"],
            "categories": [ws_s.name, 3, 0, end - 1, 0],
            "values":     [ws_s.name, 3, 1, end - 1, 1],
            "points": [
                {"fill": {"color": ADF_HIGH}},
                {"fill": {"color": ADF_MED}},
                {"fill": {"color": ADF_LOW}},
            ][:max(0, end - 3)],
        })
        chart.set_title({"name": L["chart.severity"]})
        chart.set_style(10)
        ws_s.insert_chart(2, 4, chart, {"x_scale": 1.3, "y_scale": 1.3})

    # Alerts (optional)
    if alerts:
        ws_a = wb.add_worksheet(L["sheet.alerts"])
        ws_a.right_to_left() if lang == "ar" else None
        ws_a.hide_gridlines(2)
        ws_a.merge_range("A1:D1", L["alerts.title"], fmts["section"])
        rows = [[a.get("title"), a.get("kind"), a.get("metric"), a.get("evidence", "")[:200]]
                for a in alerts]
        _write_table(ws_a, fmts,
                     [L["col.title"], L["col.kind"], L["col.metric"], L["col.evidence"]],
                     rows, start_row=2,
                     col_widths=[36, 14, 14, 60])

    wb.close()
    return buf.getvalue()


def build_patterns_workbook(
    *,
    weekly: list[dict],
    categories: list[dict],
    severity: list[dict],
    severity_weekly: Optional[list[dict]] = None,
    momentum: Optional[list[dict]] = None,
    topics: Optional[list[dict]] = None,
    weekly_by_cat: Optional[list[dict]] = None,
    subcategories: Optional[list[dict]] = None,
    filter_summary: Optional[Mapping[str, Any]] = None,
    lang: str = "ar",
) -> bytes:
    wb, buf = _new_workbook(lang)
    fmts = _formats(wb)
    L = _labels(lang)

    _add_cover_sheet(
        wb, fmts,
        title=L["patterns.title"],
        subtitle=L["patterns.subtitle"],
        filter_summary=filter_summary,
        lang=lang,
    )

    # Helper for tabular + chart sheet
    def _sheet_with_chart(sheet_name, section_title, headers, rows,
                         chart_type, x_title=None, y_title=None,
                         col_widths=None, col_formats=None,
                         chart_style="adf"):
        ws = wb.add_worksheet(sheet_name)
        ws.right_to_left() if lang == "ar" else None
        ws.hide_gridlines(2)
        ws.merge_range(0, 0, 0, max(3, len(headers) - 1), section_title, fmts["section"])
        end = _write_table(ws, fmts, headers, rows, start_row=2,
                           col_widths=col_widths, col_formats=col_formats)
        if rows:
            chart = wb.add_chart({"type": chart_type})
            chart.add_series({
                "name": section_title,
                "categories": [ws.name, 3, 0, end - 1, 0],
                "values":     [ws.name, 3, 1, end - 1, 1],
                "fill":       {"color": ADF_GREEN},
                "border":     {"color": ADF_GREEN_DARK},
                "line":       {"color": ADF_GREEN, "width": 2.25}
                              if chart_type == "line" else None,
            })
            chart.set_title({"name": section_title})
            if x_title: chart.set_x_axis({"name": x_title})
            if y_title: chart.set_y_axis({"name": y_title})
            chart.set_legend({"none": True})
            ws.insert_chart(2, max(4, len(headers) + 1), chart,
                           {"x_scale": 1.5, "y_scale": 1.4})
        return ws

    # 1. Weekly volume
    _sheet_with_chart(
        L["sheet.weekly"], L["chart.weekly"],
        [L["col.week"], L["col.count"]],
        [[r.get("week") or r.get("date"), r.get("count", 0)] for r in (weekly or [])],
        "line", x_title=L["col.week"], y_title=L["col.count"],
        col_widths=[18, 14], col_formats=["date", "num"],
    )

    # 2. Categories
    _sheet_with_chart(
        L["sheet.categories"], L["chart.categories"],
        [L["col.category"], L["col.count"]],
        [[r.get("name"), r.get("count", 0)] for r in (categories or [])],
        "bar", col_widths=[28, 14], col_formats=[None, "num"],
    )

    # 3. Severity
    _sheet_with_chart(
        L["sheet.severity"], L["chart.severity"],
        [L["col.severity"], L["col.count"]],
        [[r.get("severity"), r.get("count", 0)] for r in (severity or [])],
        "column", col_widths=[14, 14], col_formats=["sev", "num"],
    )

    # 4. Severity over weeks
    if severity_weekly:
        _sheet_with_chart(
            L["sheet.severity_weekly"], L["chart.severity_weekly"],
            [L["col.week"], L["col.high"], L["col.med"], L["col.low"]],
            [[r.get("week"), r.get("high", 0), r.get("med", 0), r.get("low", 0)]
             for r in severity_weekly],
            "line", col_widths=[18, 12, 12, 12],
            col_formats=["date", "num", "num", "num"],
        )

    # 5. Momentum
    if momentum:
        _sheet_with_chart(
            L["sheet.momentum"], L["chart.momentum"],
            [L["col.topic"], L["col.recent"], L["col.prior"], L["col.delta"]],
            [[r.get("topic"), r.get("recent", 0), r.get("prior", 0), r.get("delta", 0)]
             for r in momentum],
            "bar", col_widths=[36, 12, 12, 12],
            col_formats=[None, "num", "num", "num"],
        )

    # 6. Top topics
    if topics:
        _sheet_with_chart(
            L["sheet.topics"], L["chart.topics"],
            [L["col.topic"], L["col.count"], L["col.high"]],
            [[r.get("name"), r.get("count", 0), r.get("high_count", 0)] for r in topics],
            "bar", col_widths=[36, 12, 14],
            col_formats=[None, "num", "num"],
        )

    # 7. Subcategories
    if subcategories:
        ws = wb.add_worksheet(L["sheet.subcategories"])
        ws.right_to_left() if lang == "ar" else None
        ws.hide_gridlines(2)
        ws.merge_range("A1:D1", L["chart.subcategories"], fmts["section"])
        rows = [[r.get("category"), r.get("subcategory"), r.get("count", 0)]
                for r in subcategories]
        _write_table(ws, fmts,
                     [L["col.category"], L["col.subcategory"], L["col.count"]],
                     rows, start_row=2, col_widths=[24, 36, 12],
                     col_formats=[None, None, "num"])

    # 8. Weekly by category (long format, useful for pivots)
    if weekly_by_cat:
        ws = wb.add_worksheet(L["sheet.weekly_cat"])
        ws.right_to_left() if lang == "ar" else None
        ws.hide_gridlines(2)
        ws.merge_range("A1:D1", L["chart.weekly_cat"], fmts["section"])
        rows = [[r.get("week"), r.get("category"), r.get("count", 0)] for r in weekly_by_cat]
        _write_table(ws, fmts,
                     [L["col.week"], L["col.category"], L["col.count"]],
                     rows, start_row=2, col_widths=[18, 24, 12],
                     col_formats=["date", None, "num"])

    wb.close()
    return buf.getvalue()


def build_recommendations_workbook(
    *,
    snapshot: Optional[Mapping[str, Any]],
    insights: list[dict],
    kpis: Optional[Mapping[str, Any]] = None,
    filter_summary: Optional[Mapping[str, Any]] = None,
    lang: str = "ar",
) -> bytes:
    wb, buf = _new_workbook(lang)
    fmts = _formats(wb)
    L = _labels(lang)

    snap_meta = {}
    if snapshot:
        snap_meta = {
            L["snap.col.id"]:       snapshot.get("id", "—"),
            L["snap.col.created"]:  snapshot.get("created_at", "—"),
            L["snap.col.trigger"]:  snapshot.get("trigger", "—"),
            L["snap.col.provider"]: snapshot.get("provider", "—"),
            L["snap.col.lang"]:     snapshot.get("language", lang),
            L["snap.col.locked"]:   (L["snap.locked"] if snapshot.get("locked")
                                     else L["snap.unlocked"]),
            L["snap.col.rows"]:     snapshot.get("rows", "—"),
        }

    _add_cover_sheet(
        wb, fmts,
        title=L["snap.title"],
        subtitle=L["snap.subtitle"],
        filter_summary={**(filter_summary or {}), **snap_meta},
        kpis=kpis,
        lang=lang,
    )

    # Recommendations table
    ws = wb.add_worksheet(L["sheet.recommendations"])
    ws.right_to_left() if lang == "ar" else None
    ws.hide_gridlines(2)
    ws.merge_range("A1:F1", L["snap.title"], fmts["section"])
    headers = [L["col.kind"], L["col.title"], L["col.metric"],
               L["col.evidence"], L["col.action"], L["col.severity"]]
    rows = []
    for ins in insights or []:
        rows.append([
            ins.get("kind", ""),
            ins.get("title", ""),
            ins.get("metric", ""),
            (ins.get("evidence") or "")[:500],
            (ins.get("action") or "")[:300],
            ins.get("severity", ""),
        ])
    _write_table(ws, fmts, headers, rows, start_row=2,
                 col_widths=[14, 36, 18, 60, 50, 12],
                 col_formats=[None, None, None, None, None, "sev"])

    # Insights by kind summary chart
    if insights:
        kinds: dict[str, int] = {}
        for ins in insights:
            k = ins.get("kind") or "—"
            kinds[k] = kinds.get(k, 0) + 1
        ws_k = wb.add_worksheet(L["sheet.recommendations_by_kind"])
        ws_k.right_to_left() if lang == "ar" else None
        ws_k.hide_gridlines(2)
        ws_k.merge_range("A1:D1", L["chart.kinds"], fmts["section"])
        end = _write_table(ws_k, fmts, [L["col.kind"], L["col.count"]],
                           list(kinds.items()), start_row=2,
                           col_widths=[18, 12], col_formats=[None, "num"])
        chart = wb.add_chart({"type": "doughnut"})
        chart.add_series({
            "name": L["chart.kinds"],
            "categories": [ws_k.name, 3, 0, end - 1, 0],
            "values":     [ws_k.name, 3, 1, end - 1, 1],
        })
        chart.set_title({"name": L["chart.kinds"]})
        chart.set_style(10)
        ws_k.insert_chart(2, 4, chart, {"x_scale": 1.3, "y_scale": 1.3})

    wb.close()
    return buf.getvalue()


def build_tickets_workbook(
    *,
    df: pd.DataFrame,
    filter_summary: Optional[Mapping[str, Any]] = None,
    lang: str = "ar",
) -> bytes:
    wb, buf = _new_workbook(lang)
    fmts = _formats(wb)
    L = _labels(lang)

    total = int(len(df)) if df is not None else 0

    headline_kpis = {L["kpi.total"]: total}
    if df is not None and "severity" in df.columns and total:
        sev_counts = df["severity"].astype(str).str.lower().value_counts()
        headline_kpis[L["kpi.high"]] = int(sev_counts.get("high", 0))
        headline_kpis[L["sev.med"]] = int(sev_counts.get("med", 0))
        headline_kpis[L["sev.low"]] = int(sev_counts.get("low", 0))

    _add_cover_sheet(
        wb, fmts,
        title=L["actions.title"],
        subtitle=L["actions.subtitle"],
        filter_summary=filter_summary,
        kpis=headline_kpis,
        lang=lang,
    )

    # Tickets sheet
    ws = wb.add_worksheet(L["sheet.tickets"])
    ws.right_to_left() if lang == "ar" else None
    ws.hide_gridlines(2)
    ws.merge_range("A1:H1", L["actions.title"], fmts["section"])

    headers = [L["col.id"], L["col.category"], L["col.topic"], L["col.severity"],
               L["col.severity_reason"], L["col.action"], L["col.body"], L["col.closed"]]
    cols = ["id", "category", "topic", "severity", "severity_reason",
            "action", "body", "closed_at"]
    rows = []
    if df is not None and total:
        # Defensive column lookup
        present = [c if c in df.columns else None for c in cols]
        for _, r in df.iterrows():
            rows.append([
                (r.get(present[0]) if present[0] else ""),
                (r.get(present[1]) if present[1] else ""),
                (r.get(present[2]) if present[2] else ""),
                (r.get(present[3]) if present[3] else ""),
                (str(r.get(present[4]) or "")[:300] if present[4] else ""),
                (str(r.get(present[5]) or "")[:300] if present[5] else ""),
                (str(r.get(present[6]) or "")[:500] if present[6] else ""),
                (r.get(present[7]) if present[7] else ""),
            ])

    end = _write_table(
        ws, fmts, headers, rows, start_row=2,
        col_widths=[12, 18, 22, 12, 40, 40, 60, 16],
        col_formats=[None, None, None, "sev", None, None, None, "date"],
    )
    # Freeze header
    ws.freeze_panes(3, 0)
    # Auto filter
    if rows:
        ws.autofilter(2, 0, end - 1, len(headers) - 1)

    # Severity distribution chart
    if rows and df is not None and "severity" in df.columns:
        sev_data = df["severity"].astype(str).str.lower().value_counts()
        ws_s = wb.add_worksheet(L["sheet.severity"])
        ws_s.right_to_left() if lang == "ar" else None
        ws_s.hide_gridlines(2)
        ws_s.merge_range("A1:D1", L["chart.severity"], fmts["section"])
        sev_rows = [[k, int(v)] for k, v in sev_data.items()]
        send = _write_table(ws_s, fmts, [L["col.severity"], L["col.count"]],
                            sev_rows, start_row=2, col_widths=[14, 14],
                            col_formats=["sev", "num"])
        chart = wb.add_chart({"type": "doughnut"})
        chart.add_series({
            "name": L["chart.severity"],
            "categories": [ws_s.name, 3, 0, send - 1, 0],
            "values":     [ws_s.name, 3, 1, send - 1, 1],
            "points": [
                {"fill": {"color": ADF_HIGH}},
                {"fill": {"color": ADF_MED}},
                {"fill": {"color": ADF_LOW}},
            ][:max(0, send - 3)],
        })
        chart.set_title({"name": L["chart.severity"]})
        chart.set_style(10)
        ws_s.insert_chart(2, 4, chart, {"x_scale": 1.3, "y_scale": 1.3})

    wb.close()
    return buf.getvalue()


# =============================================================================
# Localised labels (small mirror of the front-end i18n)
# =============================================================================

_LABELS_AR = {
    "overview.title":       "نظرة عامة",
    "overview.subtitle":    "ملخّص أداء قنوات المشاركة والمؤشرات الرئيسية للفترة المختارة.",
    "patterns.title":       "الأنماط والاتجاهات",
    "patterns.subtitle":    "الموضوعات المتكرّرة، اتجاهات الخطورة، والموضوعات المتصاعدة.",
    "snap.title":           "سجل التوصيات",
    "snap.subtitle":        "أرشيف التوصيات الصادرة عن النظام مع بياناتها الوصفية.",
    "snap.locked":          "مقفلة",
    "snap.unlocked":        "مفتوحة",
    "snap.col.id":          "رقم اللقطة",
    "snap.col.created":     "تاريخ الإنشاء",
    "snap.col.trigger":     "المصدر",
    "snap.col.provider":    "المزوّد",
    "snap.col.lang":        "اللغة",
    "snap.col.locked":      "الحالة",
    "snap.col.rows":        "عدد السجلات",
    "actions.title":        "الإجراءات المقترحة",
    "actions.subtitle":     "قائمة التذاكر المُصفَّاة مع تصنيف الذكاء الاصطناعي وإجراءاتها.",
    "alerts.title":         "الإنذار المبكّر",
    "kpi.total":            "إجمالي الطلبات",
    "kpi.complaints":       "نسبة الشكاوى",
    "kpi.high":             "الخطورة العالية",
    "kpi.cats":             "الفئات النشطة",
    "kpi.insufficient":     "السياق غير الكافي",
    "sev.high":             "عالية",
    "sev.med":              "متوسطة",
    "sev.low":              "منخفضة",
    "chart.weekly":         "تطور حجم الطلبات أسبوعيًا",
    "chart.categories":     "التوزيع حسب الفئة",
    "chart.severity":       "التوزيع حسب الخطورة",
    "chart.severity_weekly":"اتجاه الخطورة عبر الأسابيع",
    "chart.momentum":       "الموضوعات المتصاعدة",
    "chart.topics":         "أعلى الموضوعات تكرارًا",
    "chart.subcategories":  "التصنيفات الفرعية",
    "chart.weekly_cat":     "الاتجاه الأسبوعي حسب الفئة",
    "chart.kinds":          "الرؤى حسب النوع",
    "col.week":             "الأسبوع",
    "col.count":            "العدد",
    "col.category":         "الفئة",
    "col.subcategory":      "التصنيف الفرعي",
    "col.severity":         "الخطورة",
    "col.severity_reason":  "تبرير الخطورة",
    "col.action":           "الإجراء المقترح",
    "col.topic":            "الموضوع",
    "col.high":             "عالية",
    "col.med":              "متوسطة",
    "col.low":              "منخفضة",
    "col.recent":           "الفترة الأخيرة",
    "col.prior":            "الفترة السابقة",
    "col.delta":            "الفرق",
    "col.id":               "رقم الطلب",
    "col.body":             "نص الطلب",
    "col.closed":           "تاريخ الإغلاق",
    "col.title":            "العنوان",
    "col.kind":             "النوع",
    "col.metric":           "المقياس",
    "col.evidence":         "الملاحظة",
    "sheet.weekly":         "الأسبوعي",
    "sheet.categories":     "الفئات",
    "sheet.severity":       "الخطورة",
    "sheet.severity_weekly":"اتجاه الخطورة",
    "sheet.momentum":       "المتصاعدة",
    "sheet.topics":         "الموضوعات",
    "sheet.subcategories":  "التصنيفات الفرعية",
    "sheet.weekly_cat":     "الأسبوعي حسب الفئة",
    "sheet.alerts":         "الإنذار المبكّر",
    "sheet.recommendations":"التوصيات",
    "sheet.recommendations_by_kind": "توزيع الرؤى",
    "sheet.tickets":        "التذاكر",
}

_LABELS_EN = {
    "overview.title":       "Overview",
    "overview.subtitle":    "Channel performance summary and key indicators for the selected period.",
    "patterns.title":       "Patterns & Trends",
    "patterns.subtitle":    "Recurring topics, severity trends, and rising themes.",
    "snap.title":           "Recommendations log",
    "snap.subtitle":        "Archive of system-issued recommendations with metadata.",
    "snap.locked":          "Locked",
    "snap.unlocked":        "Open",
    "snap.col.id":          "Snapshot ID",
    "snap.col.created":     "Created",
    "snap.col.trigger":     "Trigger",
    "snap.col.provider":    "Provider",
    "snap.col.lang":        "Language",
    "snap.col.locked":      "Status",
    "snap.col.rows":        "Rows",
    "actions.title":        "Suggested actions",
    "actions.subtitle":     "Filtered ticket list with AI classification and recommended next steps.",
    "alerts.title":         "Early warning",
    "kpi.total":            "Total requests",
    "kpi.complaints":       "Complaints share",
    "kpi.high":             "High severity",
    "kpi.cats":             "Active categories",
    "kpi.insufficient":     "Insufficient context",
    "sev.high":             "High",
    "sev.med":              "Medium",
    "sev.low":              "Low",
    "chart.weekly":         "Weekly request volume",
    "chart.categories":     "By category",
    "chart.severity":       "By severity",
    "chart.severity_weekly":"Severity over weeks",
    "chart.momentum":       "Rising topics",
    "chart.topics":         "Top recurring topics",
    "chart.subcategories":  "Subcategories",
    "chart.weekly_cat":     "Weekly by category",
    "chart.kinds":          "Insights by kind",
    "col.week":             "Week",
    "col.count":            "Count",
    "col.category":         "Category",
    "col.subcategory":      "Subcategory",
    "col.severity":         "Severity",
    "col.severity_reason":  "Severity reason",
    "col.action":           "Recommended action",
    "col.topic":            "Topic",
    "col.high":             "High",
    "col.med":              "Medium",
    "col.low":              "Low",
    "col.recent":           "Recent",
    "col.prior":            "Prior",
    "col.delta":            "Delta",
    "col.id":               "Request ID",
    "col.body":             "Body",
    "col.closed":           "Closed at",
    "col.title":            "Title",
    "col.kind":             "Kind",
    "col.metric":           "Metric",
    "col.evidence":         "Observation",
    "sheet.weekly":         "Weekly",
    "sheet.categories":     "Categories",
    "sheet.severity":       "Severity",
    "sheet.severity_weekly":"Severity weekly",
    "sheet.momentum":       "Rising",
    "sheet.topics":         "Topics",
    "sheet.subcategories":  "Subcategories",
    "sheet.weekly_cat":     "Weekly by category",
    "sheet.alerts":         "Alerts",
    "sheet.recommendations":"Recommendations",
    "sheet.recommendations_by_kind": "Insights by kind",
    "sheet.tickets":        "Tickets",
}


def _labels(lang: str) -> dict[str, str]:
    return _LABELS_AR if (lang or "ar").lower().startswith("ar") else _LABELS_EN


def filename_for(report: str, *, lang: str = "ar") -> str:
    """Suggest a download filename for a given report kind."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    base = {
        "overview":        "ADF-overview",
        "patterns":        "ADF-patterns",
        "recommendations": "ADF-recommendations",
        "tickets":         "ADF-tickets",
    }.get(report, "ADF-report")
    return f"{base}-{stamp}.xlsx"
