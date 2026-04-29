import sys
from playwright.sync_api import sync_playwright
exe = "/sessions/vigilant-elegant-hawking/.cache/ms-playwright/chromium-1217/chrome-linux/chrome"
with sync_playwright() as pw:
    b = pw.chromium.launch(headless=True, executable_path=exe)
    p = b.new_context(viewport={"width": 1440, "height": 900}).new_page()
    errors = []
    p.on("console", lambda m: errors.append(f"{m.type}: {m.text[:160]}") if m.type in ("error","warning") else None)
    p.goto("http://127.0.0.1:8501/", wait_until="networkidle", timeout=20000)
    p.wait_for_timeout(1500)
    p.evaluate("() => document.querySelector('#nav .nav__item[data-page=\"patterns\"]').click()")
    p.wait_for_timeout(2000)
    has_chart = p.evaluate("() => !!document.querySelector('#chart-subcategories')")
    has_data = p.evaluate("() => { const el = document.querySelector('#chart-subcategories'); return el && (el.innerHTML.includes('plotly') || el.querySelector('.empty')); }")
    coverage = p.evaluate("() => document.querySelector('#subcat-coverage') ? document.querySelector('#subcat-coverage').innerText : null")
    print(f"chart-subcategories present: {has_chart}")
    print(f"chart has data or fallback: {has_data}")
    print(f"coverage hint: {coverage!r}")
    p.screenshot(path="/tmp/adf/test-runs/screenshots/10_subcategories.png", full_page=True)
    if errors:
        print("\nConsole errors/warnings:")
        for e in errors[:5]: print(f"  {e}")
    b.close()
