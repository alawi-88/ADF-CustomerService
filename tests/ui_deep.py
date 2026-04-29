"""Deeper usability — drill-downs, navigation, contrast, accessibility."""
import json, sys
from playwright.sync_api import sync_playwright

exe = "/sessions/vigilant-elegant-hawking/.cache/ms-playwright/chromium-1217/chrome-linux/chrome"
issues, notes = [], []
def L(m): print(m, flush=True); notes.append(m)

with sync_playwright() as pw:
    b = pw.chromium.launch(headless=True, executable_path=exe)
    ctx = b.new_context(viewport={"width": 1440, "height": 900})
    page = ctx.new_page()

    page.goto("http://127.0.0.1:8501/", wait_until="networkidle", timeout=20000)
    page.wait_for_timeout(1500)

    # 1. Click first KPI card to test drill-down drawer
    L("=== drill-down: clicking first KPI card ===")
    clicked = page.evaluate("""() => {
        const k = document.querySelector('.kpi.drillable');
        if (!k) return false;
        k.click();
        return true;
    }""")
    L(f"  clicked KPI: {clicked}")
    page.wait_for_timeout(700)
    drawer = page.evaluate("""() => {
        const d = document.querySelector('.drawer, .side-drawer, [class*=drawer]');
        if (!d) return null;
        const cs = getComputedStyle(d);
        return {
            cls: d.className,
            visible: cs.display !== 'none' && cs.visibility !== 'hidden' && parseFloat(cs.opacity) > 0,
            transform: cs.transform.slice(0, 80),
        };
    }""")
    L(f"  drawer state: {drawer}")
    if not drawer or not drawer.get("visible"):
        issues.append(f"drill-down drawer did not open or is invisible: {drawer}")
    page.screenshot(path="/tmp/adf/test-runs/screenshots/03_drilldown.png", full_page=False)

    # Close any open drawer
    page.keyboard.press("Escape")
    page.wait_for_timeout(400)

    # 2. Try navigating to other pages — find tabs/nav
    L("=== navigation: discover tabs ===")
    nav = page.evaluate("""() => Array.from(document.querySelectorAll('.nav-item, .tab, [role=tab], .menu-item, .sidebar a, [data-page]')).map(e => ({
        cls: e.className.slice(0,80),
        txt: (e.innerText||'').trim().slice(0,40),
        target: e.getAttribute('data-page') || e.getAttribute('href') || ''
    })).filter(n => n.txt)""")
    L(f"  found {len(nav)} nav-like elements")
    for n in nav[:8]:
        L(f"    - {n['txt']!r} -> {n['target']!r} | {n['cls']}")

    # 3. Click each nav item if available, screenshot
    if nav:
        for i, n in enumerate(nav[:5]):
            try:
                # Click by text since selectors might collide
                page.evaluate(f"""(t) => {{
                    const e = Array.from(document.querySelectorAll('.nav-item, .tab, [role=tab], .menu-item, .sidebar a, [data-page]'))
                        .find(x => (x.innerText||'').trim() === t);
                    if (e) e.click();
                }}""", n["txt"])
                page.wait_for_timeout(900)
                page.screenshot(path=f"/tmp/adf/test-runs/screenshots/04_nav_{i:02d}_{n['txt'][:20].replace(' ','_').replace('/','_')}.png", full_page=False)
                L(f"  nav {i}: {n['txt']!r} ok")
            except Exception as e:
                issues.append(f"nav click {n['txt']!r} failed: {e}")

    # 4. Color contrast check on key text
    L("=== contrast checks ===")
    contrasts = page.evaluate("""() => {
        function rel(c) {
            const m = c.match(/rgba?\\(([^)]+)\\)/); if (!m) return null;
            let [r,g,b] = m[1].split(',').slice(0,3).map(s => parseFloat(s.trim()));
            [r,g,b] = [r,g,b].map(v => { v/=255; return v<=0.03928 ? v/12.92 : Math.pow((v+0.055)/1.055, 2.4); });
            return 0.2126*r + 0.7152*g + 0.0722*b;
        }
        function ratio(a, b) {
            const la = rel(a), lb = rel(b);
            if (la == null || lb == null) return null;
            const [hi, lo] = la>lb?[la,lb]:[lb,la];
            return (hi+0.05)/(lo+0.05);
        }
        const samples = [];
        const sels = ['.kpi__value', '.kpi__label', '.btn-primary', 'h1', 'h2', '.tag, .chip, .badge'];
        for (const s of sels) {
            const e = document.querySelector(s);
            if (!e) continue;
            const cs = getComputedStyle(e);
            // bubble up bg
            let p = e, bg = cs.backgroundColor;
            while (p && (bg === 'rgba(0, 0, 0, 0)' || bg === 'transparent')) {
                p = p.parentElement; if (!p) break;
                bg = getComputedStyle(p).backgroundColor;
            }
            samples.push({
                sel: s, fg: cs.color, bg, ratio: ratio(cs.color, bg),
                font: cs.fontSize + ' ' + cs.fontWeight,
                txt: (e.innerText||'').slice(0,40).trim()
            });
        }
        return samples;
    }""")
    for s in contrasts:
        ok = (s["ratio"] is None) or (s["ratio"] >= 4.5) or (s["ratio"] >= 3.0 and "px" in s["font"] and float(s["font"].split("px")[0]) >= 18)
        L(f"  {s['sel']:<20} ratio={s['ratio']!s:>8} fg={s['fg']:<20} bg={s['bg']:<20} {s['font']}  txt={s['txt']!r}")
        if s["ratio"] is not None and s["ratio"] < 4.5:
            issues.append(f"contrast {s['ratio']:.2f}:1 on {s['sel']!r} (text {s['txt']!r}) — below WCAG AA 4.5:1")

    # 5. Keyboard nav — count visible focus rings on first 8 tab stops
    L("=== keyboard: 8 tab stops, count visible focus rings ===")
    visible_focus = 0
    for i in range(8):
        page.keyboard.press("Tab")
        page.wait_for_timeout(120)
        v = page.evaluate("""() => {
            const a = document.activeElement;
            if (!a || a === document.body) return false;
            const cs = getComputedStyle(a);
            return cs.outlineStyle !== 'none' && parseFloat(cs.outlineWidth) > 0;
        }""")
        if v: visible_focus += 1
    L(f"  visible focus on {visible_focus}/8 stops")
    if visible_focus < 6:
        issues.append(f"only {visible_focus}/8 tab stops show a visible focus outline")

    # 6. Check all <img> have alt (or are decorative)
    no_alt = page.evaluate("""() => Array.from(document.querySelectorAll('img')).filter(i => !i.hasAttribute('alt')).map(i => i.src.slice(0,100))""")
    L(f"  imgs without alt: {len(no_alt)}")
    if no_alt:
        issues.append(f"{len(no_alt)} <img> without alt attribute (sample: {no_alt[:3]})")

    # 7. Check buttons without accessible name
    nameless_btns = page.evaluate("""() => {
        return Array.from(document.querySelectorAll('button')).filter(b => {
            const t = (b.innerText || b.getAttribute('aria-label') || '').trim();
            return !t;
        }).map(b => b.outerHTML.slice(0, 120));
    }""")
    L(f"  nameless buttons: {len(nameless_btns)}")
    if nameless_btns:
        issues.append(f"{len(nameless_btns)} <button> without text or aria-label (sample: {nameless_btns[:2]})")

    b.close()

print("\n=== ISSUES ===")
if not issues:
    print("(none)")
else:
    for i, it in enumerate(issues, 1):
        print(f"  {i}. {it}")

import json
from pathlib import Path
Path("/tmp/adf/test-runs/ui_deep_issues.json").write_text(json.dumps({"issues": issues, "notes": notes}, ensure_ascii=False, indent=2))
sys.exit(0 if not issues else 1)
