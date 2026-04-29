"""Render at three common viewport widths and capture screenshots + layout issues."""
import sys
from playwright.sync_api import sync_playwright
exe = "/sessions/vigilant-elegant-hawking/.cache/ms-playwright/chromium-1217/chrome-linux/chrome"

VIEWPORTS = [
    ("mobile_375", 375, 812),
    ("tablet_768", 768, 1024),
    ("laptop_1024", 1024, 768),
]

issues = []
with sync_playwright() as pw:
    b = pw.chromium.launch(headless=True, executable_path=exe)
    for label, w, h in VIEWPORTS:
        ctx = b.new_context(viewport={"width": w, "height": h}, device_scale_factor=2)
        page = ctx.new_page()
        page.goto("http://127.0.0.1:8501/", wait_until="networkidle", timeout=20000)
        page.wait_for_timeout(1500)

        # 1) Horizontal scrollbar?
        h_scroll = page.evaluate("() => document.documentElement.scrollWidth > document.documentElement.clientWidth + 1")
        print(f"[{label} {w}x{h}] horizontal scroll: {h_scroll}")
        if h_scroll: issues.append(f"{label}: page has horizontal overflow")

        # 2) Are the KPI cards stacking or running off-screen?
        kpis = page.evaluate("""() => Array.from(document.querySelectorAll('#kpi-grid .kpi')).map(el => {
            const r = el.getBoundingClientRect();
            return { x: r.x, y: r.y, w: r.width, right: r.right };
        })""")
        right_max = max((k["right"] for k in kpis), default=0)
        print(f"[{label}] kpi rightmost edge: {right_max:.0f}px (viewport: {w}px)")
        if right_max > w + 1: issues.append(f"{label}: KPI cards exceed viewport width by {right_max - w:.0f}px")

        # 3) Sidebar visible/collapsed?
        sidebar = page.evaluate("""() => {
            const s = document.querySelector('.sidebar, .nav-sidebar, aside');
            if (!s) return null;
            const r = s.getBoundingClientRect();
            return { visible: r.width > 10 && getComputedStyle(s).display !== 'none', w: r.width, x: r.x };
        }""")
        print(f"[{label}] sidebar: {sidebar}")

        # 4) Main content overflow check
        main = page.evaluate("""() => {
            const m = document.querySelector('main, .main, #main');
            if (!m) return null;
            const r = m.getBoundingClientRect();
            return { x: r.x, w: r.width, right: r.right, scrollW: m.scrollWidth, clientW: m.clientWidth };
        }""")
        print(f"[{label}] main: {main}")

        # 5) Charts — do they overflow their container?
        chart_overflows = page.evaluate("""() => {
            const overflows = [];
            document.querySelectorAll('.card .chart, [id*=chart], .plotly').forEach(el => {
                if (el.scrollWidth > el.clientWidth + 1) {
                    overflows.push({ id: el.id || el.className, scroll: el.scrollWidth, client: el.clientWidth });
                }
            });
            return overflows.slice(0, 5);
        }""")
        if chart_overflows:
            print(f"[{label}] chart overflows: {chart_overflows}")
            issues.append(f"{label}: {len(chart_overflows)} charts overflow their containers")

        page.screenshot(path=f"/tmp/adf/test-runs/screenshots/09_responsive_{label}.png", full_page=True)
        ctx.close()
    b.close()

print("\n=== RESPONSIVE ISSUES ===")
if not issues: print("(none)")
else:
    for i, x in enumerate(issues, 1): print(f"  {i}. {x}")
sys.exit(0 if not issues else 1)
