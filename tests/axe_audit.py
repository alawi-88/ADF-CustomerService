"""Inject axe-core into each page of the dashboard and report violations."""
import json, sys
from playwright.sync_api import sync_playwright

exe = "/sessions/vigilant-elegant-hawking/.cache/ms-playwright/chromium-1217/chrome-linux/chrome"
AXE_JS = "https://cdnjs.cloudflare.com/ajax/libs/axe-core/4.10.0/axe.min.js"

PAGES_TO_AUDIT = ["overview", "patterns", "alerts", "actions", "qa", "data", "snapshots"]
results_by_page = {}

with sync_playwright() as pw:
    b = pw.chromium.launch(headless=True, executable_path=exe)
    ctx = b.new_context(viewport={"width": 1440, "height": 900})
    page = ctx.new_page()
    page.goto("http://127.0.0.1:8501/", wait_until="networkidle", timeout=20000)
    page.wait_for_timeout(2000)
    page.add_script_tag(url=AXE_JS)
    page.wait_for_timeout(500)

    for nav_target in PAGES_TO_AUDIT:
        # click the nav item
        clicked = page.evaluate(f"""() => {{
            const e = document.querySelector('#nav .nav__item[data-page="{nav_target}"]');
            if (e) {{ e.click(); return true; }}
            return false;
        }}""")
        page.wait_for_timeout(1200)

        # run axe
        try:
            r = page.evaluate("""async () => {
                const result = await axe.run(document, {
                    runOnly: { type: 'tag', values: ['wcag2a', 'wcag2aa', 'wcag21aa', 'best-practice'] }
                });
                return {
                    violations: result.violations.map(v => ({
                        id: v.id,
                        impact: v.impact,
                        help: v.help,
                        helpUrl: v.helpUrl,
                        nodes: v.nodes.slice(0, 3).map(n => ({
                            target: n.target,
                            html: (n.html||'').slice(0, 200),
                            failureSummary: (n.failureSummary||'').slice(0, 200)
                        }))
                    })),
                    passes: result.passes.length,
                    incomplete: result.incomplete.length
                };
            }""")
        except Exception as e:
            r = {"error": str(e)}
        results_by_page[nav_target] = r

    b.close()

# Pretty-print summary
print("\n=== AXE-CORE AUDIT ===\n")
for page_name, r in results_by_page.items():
    if "error" in r:
        print(f"[{page_name}] ERROR: {r['error']}")
        continue
    print(f"[{page_name}] passes={r['passes']}  incomplete={r['incomplete']}  violations={len(r['violations'])}")
    for v in r["violations"]:
        print(f"   {(v['impact'] or '?').upper():<10} {v['id']:<35} {v['help']}")
        for n in v["nodes"]:
            tgt = n["target"][0] if n["target"] else ""
            print(f"      target={tgt}  html={n['html'][:80]}")

# Tally severity across pages
from collections import Counter
all_violations = []
for r in results_by_page.values():
    if "violations" in r:
        for v in r["violations"]:
            all_violations.append((v["id"], v["impact"]))
sev = Counter(v[1] for v in all_violations)
ids = Counter(v[0] for v in all_violations)
print("\n=== SEVERITY TALLY ===")
for k in ["critical", "serious", "moderate", "minor", None]:
    if sev.get(k): print(f"  {k}: {sev[k]}")
print("\n=== UNIQUE VIOLATION IDs (any page) ===")
for k, n in ids.most_common():
    print(f"  ({n})  {k}")

import json
from pathlib import Path
Path("/tmp/adf/test-runs/axe_results.json").write_text(json.dumps(results_by_page, indent=2, ensure_ascii=False))
