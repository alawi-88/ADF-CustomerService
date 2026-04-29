import json, sys, traceback
from playwright.sync_api import sync_playwright

issues = []
notes = []

def log(msg):
    print(msg, flush=True)
    notes.append(msg)

with sync_playwright() as pw:
    browser = pw.chromium.launch(headless=True, executable_path="/sessions/vigilant-elegant-hawking/.cache/ms-playwright/chromium-1217/chrome-linux/chrome")
    ctx = browser.new_context(viewport={"width": 1440, "height": 900})

    # Capture console errors
    def on_console(m):
        if m.type in ("error", "warning"):
            issues.append(f"console.{m.type}: {m.text[:200]}")
    ctx.on("console", on_console)

    page = ctx.new_page()
    try:
        resp = page.goto("http://127.0.0.1:8501/", wait_until="networkidle", timeout=20000)
        log(f"GET / -> {resp.status if resp else 'no-response'}")
    except Exception as e:
        log(f"navigation FAILED: {e}")
        sys.exit(1)

    page.wait_for_timeout(1500)

    # 1. dir attribute
    direction = page.evaluate("() => document.documentElement.getAttribute('dir')")
    log(f"<html dir>={direction}")
    if direction != "rtl":
        issues.append(f"html dir is '{direction}', expected 'rtl' on first load")

    # 2. lang
    lang = page.evaluate("() => document.documentElement.lang")
    log(f"<html lang>={lang}")

    # 3. dga_overrides loaded?
    has_override = page.evaluate("""() => Array.from(document.styleSheets).some(s => (s.href||'').includes('dga_overrides'))""")
    log(f"dga_overrides loaded: {has_override}")
    if not has_override:
        issues.append("dga_overrides.css link missing from <head>")

    # 4. skip-link present?
    skip = page.evaluate("""() => { const a = document.querySelector('.skip-link'); return a ? a.outerHTML : null }""")
    log(f"skip-link: {skip[:120] if skip else 'MISSING'}")
    if not skip:
        issues.append("skip-link <a class='skip-link'> not found")

    # 5. KPI cards exist + render numbers
    kpis = page.evaluate("""() => Array.from(document.querySelectorAll('.kpi-card, [class*=kpi]')).slice(0,8).map(e => ({
        cls: e.className.slice(0,80),
        txt: e.innerText.slice(0,80).replace(/\\s+/g,' ').trim()
    }))""")
    log(f"KPI-like cards found: {len(kpis)}")
    for k in kpis[:6]:
        log(f"  - {k['cls']} | {k['txt']}")
    if not kpis:
        issues.append("no KPI-like elements found")

    # 6. Primary color check — sample a button
    primary_color = page.evaluate("""() => {
        const v = getComputedStyle(document.documentElement).getPropertyValue('--color-primary');
        return v ? v.trim() : null;
    }""")
    log(f"--color-primary computed: {primary_color}")
    if primary_color and primary_color.upper() != "#006C35":
        issues.append(f"--color-primary is {primary_color}, DGA expects #006C35")

    # 7. Focus state — tab to first focusable
    page.keyboard.press("Tab")
    page.wait_for_timeout(300)
    focus_outline = page.evaluate("""() => {
        const a = document.activeElement;
        if (!a || a === document.body) return null;
        const cs = getComputedStyle(a);
        return { tag: a.tagName, cls: a.className.slice(0,80),
                 outline: cs.outlineStyle + ' ' + cs.outlineWidth + ' ' + cs.outlineColor,
                 visible: cs.outlineStyle !== 'none' && parseFloat(cs.outlineWidth) > 0 };
    }""")
    log(f"after Tab — active element: {focus_outline}")
    if focus_outline and not focus_outline.get("visible"):
        issues.append(f"first tab-stop has no visible focus outline ({focus_outline})")

    # 8. Screenshot full page
    page.screenshot(path="/tmp/adf/test-runs/screenshots/01_default.png", full_page=True)
    log("screenshot saved: 01_default.png")

    # 9. Try clicking a language toggle — search likely candidates
    toggle_clicked = page.evaluate("""() => {
        const cands = Array.from(document.querySelectorAll('button, a, [role=button]')).filter(e => {
            const t = (e.innerText || '').trim().toLowerCase();
            return t === 'en' || t === 'english' || t === 'العربية' || t === 'إنجليزي' || t === 'ar';
        });
        if (cands.length) { cands[0].click(); return cands[0].innerText.trim(); }
        return null;
    }""")
    log(f"language-toggle click attempt: {toggle_clicked!r}")
    if toggle_clicked:
        page.wait_for_timeout(800)
        new_dir = page.evaluate("() => document.documentElement.getAttribute('dir')")
        new_lang = page.evaluate("() => document.documentElement.lang")
        log(f"after toggle: dir={new_dir}, lang={new_lang}")
        page.screenshot(path="/tmp/adf/test-runs/screenshots/02_toggled.png", full_page=True)

    # 10. Check that #main exists as the skip-link target
    main_target = page.evaluate("""() => {
        const m = document.getElementById('main');
        return m ? { tag: m.tagName, vis: m.offsetParent !== null } : null;
    }""")
    log(f"#main target: {main_target}")
    if not main_target:
        issues.append("#main target missing — skip-link points to nothing")

    browser.close()

# Verdict
print("\n=== ISSUES FOUND ===")
if not issues:
    print("(none)")
else:
    for i, it in enumerate(issues, 1):
        print(f"  {i}. {it}")

import json
from pathlib import Path
Path("/tmp/adf/test-runs/ui_issues.json").write_text(json.dumps({"issues": issues, "notes": notes}, ensure_ascii=False, indent=2))
sys.exit(0 if not issues else 1)
