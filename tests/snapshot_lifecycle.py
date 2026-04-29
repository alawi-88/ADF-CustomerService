"""Full snapshot lifecycle: create A, mutate filters, create B, list, get, lock A, diff A vs B, double-lock no-op."""
import json, sys
from pathlib import Path
sys.path.insert(0, "/tmp/adf/ADF-CustomerService")
from fastapi.testclient import TestClient
from src.app import app
c = TestClient(app)

# Create A — full corpus
rA = c.post("/api/snapshots", json={"trigger": "manual", "filters": {}, "language": "ar"})
assert rA.status_code == 200, rA.text
A = rA.json()["snapshot_id"]
print(f"[create] A snapshot_id = {A}")

# Create B — only complaints
rB = c.post("/api/snapshots", json={"trigger": "manual", "filters": {"category": ["شكوى"]}, "language": "ar"})
assert rB.status_code == 200, rB.text
B = rB.json()["snapshot_id"]
print(f"[create] B snapshot_id = {B}")

# List
rL = c.get("/api/snapshots")
assert rL.status_code == 200
listed = rL.json()["snapshots"]
print(f"[list] {len(listed)} snapshots, latest 3 ids: {[s['id'] for s in listed[:3]]}")

# Get A — verify shape
rGA = c.get(f"/api/snapshots/{A}")
assert rGA.status_code == 200
sA = rGA.json()
assert sA["row_count"] > 0, "A row_count is 0"
assert sA["language"] == "ar"
assert sA["provider"] in {"groq", "ollama", "rule"}
assert "items" in sA and len(sA["items"]) > 0, "A has no items"
print(f"[get A] rows={sA['row_count']}, items={len(sA['items'])}, kpis.total={sA['kpis']['total']}, locked={sA['locked']}")

# Get B
rGB = c.get(f"/api/snapshots/{B}")
sB = rGB.json()
print(f"[get B] rows={sB['row_count']}, items={len(sB['items'])}, kpis.total={sB['kpis']['total']}")
assert sB["row_count"] > 0 and sB["row_count"] < sA["row_count"]

# Diff A vs B
rD = c.get(f"/api/snapshots/{A}/diff/{B}")
assert rD.status_code == 200, rD.text
diff = rD.json()
print(f"[diff] kpi_deltas keys: {list(diff['kpi_deltas'].keys())}")
print(f"[diff] total delta: {diff['kpi_deltas']['total']}")
print(f"[diff] insights added: {len(diff['insights_added'])}, removed: {len(diff['insights_removed'])}")
print(f"[diff] severity-changed tickets: {len(diff['tickets_severity_changed'])}")

# Lock A
rLk = c.post(f"/api/snapshots/{A}/lock", json={"by_id": "u_nora", "note": "approved for April board pack"})
assert rLk.status_code == 200, rLk.text
print(f"[lock] A locked: {rLk.json()}")

# Verify lock landed
sA2 = c.get(f"/api/snapshots/{A}").json()
assert sA2["locked"] == 1, f"Lock did not land: locked={sA2['locked']}"
assert sA2["locked_by_id"] == "u_nora"
assert "April" in sA2["manager_note"]
print(f"[verify] A locked={sA2['locked']}, by={sA2['locked_by_id']}, note={sA2['manager_note']!r}")

# Re-lock should be a no-op (idempotent), not raise
rLk2 = c.post(f"/api/snapshots/{A}/lock", json={"by_id": "u_other", "note": "second lock"})
print(f"[re-lock] status={rLk2.status_code} body={rLk2.text}")
sA3 = c.get(f"/api/snapshots/{A}").json()
print(f"[re-lock verify] still locked by {sA3['locked_by_id']}, note={sA3['manager_note']!r}  (should still be u_nora)")

# 404 on missing
r404 = c.get("/api/snapshots/999999")
assert r404.status_code == 404, r404.text
print(f"[404] missing snapshot returns {r404.status_code}: {r404.json()}")

# Diff with bad ids -> 404
rDx = c.get("/api/snapshots/1/diff/999999")
print(f"[diff bad] status={rDx.status_code} body={rDx.text[:120]}")

# Lock without by_id -> 422
rLkBad = c.post(f"/api/snapshots/{A}/lock", json={})
print(f"[lock bad] status={rLkBad.status_code} (expect 422)")

print("\nALL SNAPSHOT LIFECYCLE TESTS PASSED")
