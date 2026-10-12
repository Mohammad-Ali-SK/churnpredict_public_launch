"""Smoke-test ChurnPredict's chunked CSV route via FastAPI TestClient.
Run from repository root: python churnpredict_streaming_smoke_test.py
This generates a temporary 12,000-row synthetic CSV in memory and does not modify project files.
"""
import csv
import io
import sys

from fastapi.testclient import TestClient
from app.main import app

ROWS = 12_000
columns = [
    "CustomerID", "TenureMonths", "MonthlyCharges",
    "Contract", "InternetService", "PaymentMethod", "Churn",
]
contracts = ["Month-to-month", "One year", "Two year"]
internet = ["Fiber optic", "DSL", "No"]
payments = ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]

buffer = io.StringIO(newline="")
writer = csv.writer(buffer)
writer.writerow(columns)
for i in range(1, ROWS + 1):
    tenure = (i * 7) % 73
    contract = contracts[i % 3]
    churn = "Yes" if (tenure < 18 and contract == "Month-to-month") or (i % 7 == 0 and i % 2 == 1) else "No"
    writer.writerow([
        100_000 + i,
        tenure,
        round(20 + (i * 13) % 80, 2),
        contract,
        internet[i % 3],
        payments[i % 4],
        churn,
    ])
content = buffer.getvalue().encode("utf-8")
print(f"Generated synthetic CSV: {ROWS:,} rows x {len(columns)} columns, {len(content):,} bytes")

with TestClient(app) as client:
    response = client.post(
        "/api/analyze-smart",
        files={"file": ("streaming_smoke.csv", content, "text/csv")},
        data={"target_column": "Churn", "positive_label": "Yes"},
    )

print("HTTP status:", response.status_code)
if response.status_code != 200:
    print(response.text[:3000])
    sys.exit(1)

result = response.json()
dataset = result.get("dataset", {})
validation = result.get("cross_validation", {})
risk = result.get("risk_summary", {})
checks = {
    "status is ok": result.get("status") == "ok",
    "all rows counted": dataset.get("rows") == ROWS,
    "column count correct": dataset.get("columns") == len(columns),
    "holdout evaluation reported": "holdout" in str(validation.get("method", "")).lower(),
    "training sample capped at 5,000": validation.get("training_sample_rows") == 5_000,
    "full dataset risk scan completed": risk.get("total") == ROWS,
    "detailed predictions capped at 1,000": result.get("prediction_count_returned") == 1_000,
    "shared model not saved": result.get("model_saved") is False,
}
print("\nCHECKS")
for name, passed in checks.items():
    print(f"{'PASS' if passed else 'FAIL'}: {name}")
print("Selected model:", result.get("selected_model"))
print("Training sample rows:", validation.get("training_sample_rows"))
print("Predictions returned:", result.get("prediction_count_returned"))
print("Risk summary:", risk)
if not all(checks.values()):
    sys.exit(2)
print("\nSUCCESS: Chunked CSV endpoint smoke test passed.")
