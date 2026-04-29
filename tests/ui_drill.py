"""Investigate the drill-down behaviour properly."""
from playwright.sync_api import sync_playwright
exe = "/sessions/vigilant-elegant-hawking/.cache/ms-playwright/chromium-1217/chrome-linux/chrome"
with sync_playwright() as pw:
    b = pw.chromium.launch(headless=True, executable_path=exe)
    ctx = b.new_context(viewport={"width": 1440, "height": 900})
    page = ctx.new_page()
    page.goto("http://127.0.0.1:8501/", wait_until="networkidle", timeout=20000)
    page.wait_for_timeout(1500)

    # Find the KPI and check what changes happen on click
    before = page.evaluate("""() => {
        return {
            visible_panels: Array.from(document.querySelectorAll('[class*=panel], [class*=drawer], [class*=modal], [class*=sheet], dialog'))
                .filter(e => { const cs = getComputedStyle(e); return cs.display !== 'none' && cs.visibility !== 'hidden' && parseFloat(cs.opacity) > 0; })
                .map(e => ({tag: e.tagName, cls: e.className.slice(0,80)})),
        };
    }""")
    print(f"BEFORE click: {before}")

    page.evaluate("() => document.querySelector('.kpi.drillable').click()")
    page.wait_for_timeout(800)

    after = page.evaluate("""() => {
        return {
            visible_panels: Array.from(document.querySelectorAll('[class*=panel], [class*=drawer], [class*=modal], [class*=sheet], dialog, [aria-modal]'))
                .filter(e => { const cs = getComputedStyle(e); return cs.display !== 'none' && cs.visibility !== 'hidden' && parseFloat(cs.opacity) > 0; })
                .map(e => ({tag: e.tagName, cls: e.className.slice(0,100), aria_modal: e.getAttribute('aria-modal'), id: e.id})),
            url: window.location.href,
            documents_drawers: Array.from(document.querySelectorAll('[class*=drawer]')).map(e => ({
                cls: e.className, display: getComputedStyle(e).display, visibility: getComputedStyle(e).visibility, opacity: getComputedStyle(e).opacity,
                pos: getComputedStyle(e).position, transform: getComputedStyle(e).transform, right: getComputedStyle(e).right, left: getComputedStyle(e).left
            })),
            list_panels: Array.from(document.querySelectorAll('.records, [class*=list], [class*=drill]')).filter(e => {
                const cs = getComputedStyle(e); return cs.display !== 'none';
            }).slice(0, 5).map(e => ({tag: e.tagName, cls: e.className.slice(0,80), rect: e.getBoundingClientRect().toJSON()}))
        };
    }""")
    import json
    print("AFTER click:")
    print(json.dumps(after, indent=2, default=str)[:2500])

    page.screenshot(path="/tmp/adf/test-runs/screenshots/05_after_kpi_click.png", full_page=True)
    b.close()
