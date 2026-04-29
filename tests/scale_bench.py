import sys, time, json
sys.path.insert(0, "/tmp/adf/ADF-CustomerService")
from fastapi.testclient import TestClient
from src.app import app
c = TestClient(app)

# Re-load the cached data
print("warming up...")
c.get("/api/health")
c.get("/api/kpis")

ENDPOINTS = [
    ("/api/health", None),
    ("/api/meta", None),
    ("/api/kpis", None),
    ("/api/categories", None),
    ("/api/severity", None),
    ("/api/weekly", None),
    ("/api/weekly_by_cat", None),
    ("/api/severity_weekly", None),
    ("/api/topics", None),
    ("/api/topic_momentum", None),
    ("/api/category_matrix", None),
    ("/api/alerts", None),
    ("/api/insights", None),
    ("/api/forecast", None),
    ("/api/recurring_cases", None),
    ("/api/records?limit=50", None),
    ("/api/records?limit=200", None),
    ("/api/records?body_contains=قرض&limit=50", None),
    ("/api/snapshots", None),
]

print(f"{'PATH':<50}{'STATUS':<8}{'P50_ms':<10}{'P95_ms':<10}{'BYTES':<10}")
print("-" * 90)
all_ok = True
slow = []
for path, _ in ENDPOINTS:
    samples = []
    last = None
    for _ in range(5):
        t0 = time.perf_counter()
        r = c.get(path)
        t1 = time.perf_counter()
        samples.append((t1 - t0) * 1000)
        last = r
    samples.sort()
    p50 = samples[len(samples)//2]
    p95 = samples[-1]
    sz = len(last.text)
    ok = 200 <= last.status_code < 300
    if not ok: all_ok = False
    if p95 > 1000: slow.append((path, p95))
    print(f"{path[:49]:<50}{last.status_code:<8}{p50:<10.1f}{p95:<10.1f}{sz:<10}")

print(f"\nALL OK: {all_ok}")
if slow:
    print("\nSLOW (P95 > 1000ms):")
    for p, t in sorted(slow, key=lambda x: -x[1]):
        print(f"  {t:>7.0f}ms  {p}")

# Snapshot the entire 30k corpus and time it — the heaviest operation
print("\n--- POST /api/snapshots (30k rows) ---")
t0 = time.perf_counter()
r = c.post("/api/snapshots", json={"trigger":"manual","filters":{},"language":"ar"})
dt = (time.perf_counter() - t0) * 1000
print(f"status={r.status_code}, time={dt:.0f}ms, snapshot_id={r.json().get('snapshot_id')}")
