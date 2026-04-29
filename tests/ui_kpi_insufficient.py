"""Verify the new Insufficient-context KPI tile renders."""
import sys
from playwright.sync_api import sync_playwright
exe = "/sessions/vigilant-elegant-hawking/.cache/ms-playwright/chromium-1217/chrome-linux/chrome"
with sync_playwright() as pw:
    b = pw.chromium.launch(headless=True, executable_path=exe)
    page = b.new_context(viewport={"width": 1440, "height": 900}).new_page()
    page.goto("http://127.0.0.1:8501/", wait_until="networkidle", timeout=20000)
    page.wait_for_timeout(2000)
    tiles = page.evaluate("""() => Array.from(document.querySelectorAll('#kpi-grid .kpi')).map(el => ({
        label: (el.querySelector('.kpi__label')||{}).innerText || '',
        value: (el.querySelector('.kpi__value')||{}).innerText || '',
        delta: (el.querySelector('.kpi__delta')||{}).innerText || '',
        drill: el.dataset.drill || ''
    }))""")
    for t in tiles:
        print(f"  {t['label']:<40} {t['value']:<12} {t['delta']:<60} drill={t['drill']!r}")
    page.screenshot(path="/tmp/adf/test-runs/screenshots/08_insufficient_kpi.png", full_page=False)

    has_insuff = any(t["drill"] == "insufficient" for t in tiles)
    print(f"\ninsufficient tile present: {has_insuff}")
    b.close()
    sys.exit(0 if has_insuff else 1)
