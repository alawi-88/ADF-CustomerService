"""End-to-end exercise. Fixed qa shape and added snapshot lifecycle."""
import json, sys
from pathlib import Path
sys.path.insert(0, "/tmp/adf/ADF-CustomerService")
from fastapi.testclient import TestClient
from src.app import app
c = TestClient(app)

ENDPOINTS = [
    ("GET", "/api/health", None),
    ("GET", "/api/meta", None),
    ("GET", "/api/kpis", None),
    ("GET", "/api/categories", None),
    ("GET", "/api/severity", None),
    ("GET", "/api/weekly", None),
    ("GET", "/api/weekly_by_cat", None),
    ("GET", "/api/severity_weekly", None),
    ("GET", "/api/topics", None),
    ("GET", "/api/topic_momentum", None),
    ("GET", "/api/category_matrix", None),
    ("GET", "/api/alerts", None),
    ("GET", "/api/insights", None),
    ("GET", "/api/forecast", None),
    ("GET", "/api/recurring_cases", None),
    ("GET", "/api/records?limit=5", None),
    ("GET", "/api/files", None),
    ("GET", "/api/snapshots", None),
    ("POST", "/api/snapshots", {"trigger": "manual", "filters": {}, "language": "ar"}),
    ("POST", "/api/qa", {"question": "ما أبرز الشكاوى هذا الأسبوع؟", "language": "ar"}),
    # Filter combinations
    ("GET", "/api/kpis?from=2026-04-01&to=2026-04-26", None),
    ("GET", "/api/kpis?category=شكوى", None),
    ("GET", "/api/records?limit=3&body_contains=قرض", None),
    ("GET", "/api/records?only_high=true&limit=3", None),
]

results = []
for method, path, payload in ENDPOINTS:
    try:
        r = c.request(method, path, json=payload, timeout=30)
        try:
            j = r.json()
            shape = (
                f"keys={list(j.keys())[:8]}" if isinstance(j, dict)
                else f"list_len={len(j)}" if isinstance(j, list)
                else type(j).__name__
            )
        except Exception:
            j = None
            shape = "non-json"
        results.append({
            "method": method, "path": path, "status": r.status_code,
            "size": len(r.text), "shape": shape,
            "preview": r.text[:240].replace("\n", " "),
            "ok": 200 <= r.status_code < 300,
        })
    except Exception as e:
        results.append({"method": method, "path": path, "status": "EXC",
                        "size": 0, "shape": "", "preview": f"{type(e).__name__}: {e}",
                        "ok": False})

print(f"{'M':<5}{'PATH':<48}{'STATUS':<8}{'SIZE':<10}SHAPE")
print("-" * 130)
for r in results:
    print(f"{r['method']:<5}{r['path'][:47]:<48}{str(r['status']):<8}{r['size']:<10}{r['shape']}")
print("-" * 130)
fails = [r for r in results if not r["ok"]]
print(f"OK: {len(results)-len(fails)}/{len(results)}  FAILURES: {len(fails)}")
for r in fails:
    print(f"  ! {r['method']} {r['path']} -> {r['status']}")
    print(f"    {r['preview'][:300]}")

Path("/tmp/adf/test-runs/e2e_results.json").write_text(
    json.dumps(results, ensure_ascii=False, indent=2, default=str)
)
sys.exit(0 if not fails else 1)
