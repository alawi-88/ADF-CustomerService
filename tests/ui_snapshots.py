"""Verify the new Snapshots page works end-to-end via UI clicks."""
import sys
from playwright.sync_api import sync_playwright
exe = "/sessions/vigilant-elegant-hawking/.cache/ms-playwright/chromium-1217/chrome-linux/chrome"
issues = []

with sync_playwright() as pw:
    b = pw.chromium.launch(headless=True, executable_path=exe)
    ctx = b.new_context(viewport={"width": 1440, "height": 900})
    page = ctx.new_page()
    page.on("console", lambda m: print(f"[console.{m.type}] {m.text[:200]}") if m.type in ("error","warning") else None)
    page.goto("http://127.0.0.1:8501/", wait_until="networkidle", timeout=20000)
    page.wait_for_timeout(1500)

    # 1) Nav contains snapshots
    has_nav = page.evaluate("() => !!document.querySelector('[data-page=\"snapshots\"]')")
    print(f"snapshots nav present: {has_nav}")
    if not has_nav: issues.append("snapshots nav missing")

    # 2) Click into the snapshots page
    page.evaluate("() => document.querySelector('#nav .nav__item[data-page=\"snapshots\"]').click()")
    page.wait_for_timeout(800)

    visible_section = page.evaluate("() => { const s=document.querySelector('section[data-page=\"snapshots\"]'); return s && s.classList.contains('is-visible'); }")
    print(f"snapshots section visible after click: {visible_section}")
    if not visible_section: issues.append("section did not become visible")

    # 3) Table renders (or empty-state)
    table_present = page.evaluate("() => { const t=document.querySelector('#snap-table-wrap table'); return t ? t.querySelectorAll('tbody tr').length : 0; }")
    print(f"existing snapshots in table: {table_present}")

    # 4) Click "Create snapshot now"
    page.evaluate("() => document.querySelector('#snap-create').click()")
    page.wait_for_timeout(1500)
    new_count = page.evaluate("() => { const t=document.querySelector('#snap-table-wrap table'); return t ? t.querySelectorAll('tbody tr').length : 0; }")
    print(f"after create: {new_count} rows  (should be > {table_present})")
    if new_count <= table_present:
        issues.append(f"create did not add a row ({table_present} -> {new_count})")

    # 5) Check two checkboxes and click Compare
    page.evaluate("""() => {
        const cbs = document.querySelectorAll('#snap-table-wrap input[data-snap-check]');
        if (cbs.length >= 2) {
            cbs[0].click();
            cbs[1].click();
        }
    }""")
    page.wait_for_timeout(400)
    compare_enabled = page.evaluate("() => !document.querySelector('#snap-compare').disabled")
    print(f"compare button enabled after 2 selections: {compare_enabled}")
    if not compare_enabled: issues.append("compare button still disabled with 2 selections")

    page.evaluate("() => document.querySelector('#snap-compare').click()")
    page.wait_for_timeout(1000)
    diff_visible = page.evaluate("() => { const d=document.querySelector('#snap-diff'); return d && getComputedStyle(d).display !== 'none'; }")
    diff_text = page.evaluate("() => document.querySelector('#snap-diff-body').innerText.slice(0,200)")
    print(f"diff card visible: {diff_visible}")
    print(f"diff body preview: {diff_text!r}")
    if not diff_visible: issues.append("diff card did not appear")

    # 6) Screenshot
    page.screenshot(path="/tmp/adf/test-runs/screenshots/06_snapshots_page.png", full_page=True)
    print("screenshot saved: 06_snapshots_page.png")

    # 7) Switch language and verify labels translated
    page.evaluate("() => document.querySelector('#btn-lang').click()")
    page.wait_for_timeout(800)
    nav_text = page.evaluate("() => document.querySelector('#nav .nav__item[data-page=\"snapshots\"] span').innerText")
    print(f"nav label after EN toggle: {nav_text!r}")
    if "Recommendations" not in nav_text and "log" not in nav_text.lower():
        issues.append(f"EN translation missing for snapshots: {nav_text!r}")

    page.screenshot(path="/tmp/adf/test-runs/screenshots/07_snapshots_en.png", full_page=True)
    b.close()

print("\n=== ISSUES ===")
if not issues: print("(none)")
else:
    for i, x in enumerate(issues, 1): print(f"  {i}. {x}")
sys.exit(0 if not issues else 1)
