"""Verify upload triggers an auto-snapshot with trigger=upload."""
import sys
sys.path.insert(0, "/tmp/adf/ADF-CustomerService")
from fastapi.testclient import TestClient
from src.app import app
c = TestClient(app)

# count snapshots before
before = c.get("/api/snapshots").json()["snapshots"]
print(f"snapshots before: {len(before)}")

# upload a copy of the real xlsx (renamed to avoid duplicate-skip)
src_xlsx = "/sessions/vigilant-elegant-hawking/mnt/uploads/بيانات مشاركة المستخدمين.xlsx"
with open(src_xlsx, "rb") as f:
    data = f.read()

r = c.post("/api/upload",
    files={"file": ("upload_test_v1.xlsx", data, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")})
print(f"upload status: {r.status_code}")
body = r.json()
print(f"upload body keys: {list(body.keys())}")
print(f"  saved_as: {body.get('saved_as')}")
print(f"  snapshot_id from upload: {body.get('snapshot_id')}")

# count snapshots after
after = c.get("/api/snapshots").json()["snapshots"]
print(f"snapshots after: {len(after)}")
assert len(after) == len(before) + 1, f"expected one new snapshot, got {len(after) - len(before)}"

newest = after[0]
print(f"newest snapshot: id={newest['id']}, trigger={newest['trigger']!r}, source_file={newest['source_file']!r}")
assert newest["trigger"] == "upload", f"trigger should be 'upload', got {newest['trigger']!r}"
assert newest["source_file"] == "upload_test_v1.xlsx" or newest["source_file"].startswith("upload_test_v1"), f"source_file: {newest['source_file']!r}"
assert newest["id"] == body["snapshot_id"], f"id mismatch: {newest['id']} vs {body['snapshot_id']}"

# Verify the snapshot has the same row count as the dataset
detail = c.get(f"/api/snapshots/{newest['id']}").json()
print(f"snapshot detail: rows={detail['row_count']}, items={len(detail['items'])}, kpis.total={detail['kpis']['total']}")
assert detail["row_count"] > 0
print("\nAUTO-SNAPSHOT ON UPLOAD: PASS")
