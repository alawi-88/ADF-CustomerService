from playwright.sync_api import sync_playwright
exe = "/sessions/vigilant-elegant-hawking/.cache/ms-playwright/chromium-1217/chrome-linux/chrome"
with sync_playwright() as pw:
    b = pw.chromium.launch(headless=True, executable_path=exe)
    ctx = b.new_context()
    page = ctx.new_page()
    fails = []
    page.on("response", lambda r: fails.append(f"{r.status} {r.url}") if r.status >= 400 else None)
    page.goto("http://127.0.0.1:8501/", wait_until="networkidle", timeout=20000)
    page.wait_for_timeout(1500)
    print("HTTP failures:")
    for f in fails:
        print(f"  {f}")
    # Also dump main candidates
    main_check = page.evaluate("""() => {
        return {
            id_main: !!document.getElementById('main'),
            main_tag: !!document.querySelector('main'),
            role_main: !!document.querySelector('[role=main]'),
            content_id: !!document.getElementById('content'),
            app_id: !!document.getElementById('app')
        }
    }""")
    print("Main-target candidates present:", main_check)
    b.close()
