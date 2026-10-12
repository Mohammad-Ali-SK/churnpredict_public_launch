from __future__ import annotations

import csv
import gzip
import shutil
import tempfile
import time
import warnings
from pathlib import Path

from fastapi.testclient import TestClient
from app.main import app

ROWS = 200_000
COLUMNS = [f"Feature{i:03d}" for i in range(1, 100)] + ["Churn"]
MAX_EXPANDED_BYTES = 25 * 1024 * 1024


def main() -> None:
    started = time.time()
    with tempfile.TemporaryDirectory(prefix="churnpredict-200k-test-") as temp_dir:
        temp = Path(temp_dir)
        csv_path = temp / "streaming_test_200k_100.csv"
        gzip_path = temp / "streaming_test_200k_100.csv.gz"

        print(f"Generating synthetic CSV: {ROWS:,} rows x {len(COLUMNS)} columns")
        contracts = ["A", "B", "C"]
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(COLUMNS)
            for i in range(1, ROWS + 1):
                churn = "Yes" if i % 5 == 0 or i % 11 == 0 else "No"
                dense = [int(churn == "Yes")] + [
                    (i * j + i // 3 + j) % 2 for j in range(2, 16)
                ]
                sparse = [(i + j) % 2 if i <= 2 else "" for j in range(16, 100)]
                writer.writerow(dense + sparse + [churn])

        expanded_size = csv_path.stat().st_size
        print(f"Expanded CSV size: {expanded_size:,} bytes ({expanded_size / 1024 / 1024:.2f} MiB)")
        if expanded_size > MAX_EXPANDED_BYTES:
            raise SystemExit("FAIL: Generated CSV exceeds the backend 25 MiB decompressed limit.")

        with csv_path.open("rb") as source, gzip_path.open("wb") as destination:
            with gzip.GzipFile(fileobj=destination, mode="wb", compresslevel=6) as compressor:
                shutil.copyfileobj(source, compressor, length=1024 * 1024)

        compressed_size = gzip_path.stat().st_size
        print(f"Gzipped upload size: {compressed_size:,} bytes ({compressed_size / 1024 / 1024:.2f} MiB)")

        print("Posting compressed CSV through the FastAPI test client...")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with TestClient(app) as client, gzip_path.open("rb") as compressed:
                response = client.post(
                    "/api/analyze-smart",
                    files={"file": (gzip_path.name, compressed, "application/gzip")},
                    data={"target_column": "Churn", "positive_label": "Yes"},
                )

        print("HTTP status:", response.status_code)
        if response.status_code != 200:
            print("Response:", response.text[:3000])
            raise SystemExit("FAIL: Streaming endpoint returned a non-200 status.")

        result = response.json()
        dataset = result.get("dataset", {})
        validation = result.get("cross_validation", {})
        risk = result.get("risk_summary", {})
        checks = {
            "status is ok": result.get("status") == "ok",
            "all rows counted": dataset.get("rows") == ROWS,
            "100 columns reported": dataset.get("columns") == 100,
            "holdout validation reported": "holdout" in str(validation.get("method", "")).lower(),
            "training sample capped at 5,000": 0 < validation.get("training_sample_rows", 0) <= 5_000,
            "full dataset risk scan completed": risk.get("total") == ROWS,
            "detailed predictions capped at 1,000": result.get("prediction_count_returned", ROWS + 1) <= 1_000,
            "shared model not saved": result.get("model_saved") is False,
        }
        print("\nCHECKS")
        for label, passed in checks.items():
            print(("PASS" if passed else "FAIL") + ": " + label)
        print("Selected model:", result.get("selected_model"))
        print("Training sample rows:", validation.get("training_sample_rows"))
        print("Predictions returned:", result.get("prediction_count_returned"))
        print("Risk summary:", risk)
        print("Warnings captured:", len(caught))
        for warning in caught[:5]:
            print("WARNING:", str(warning.message)[:250])
        print("Elapsed seconds:", round(time.time() - started, 1))

        if not all(checks.values()):
            raise SystemExit("FAIL: One or more 200k streaming checks failed.")
        print("\nSUCCESS: 200,000-row chunked streaming test passed.")


if __name__ == "__main__":
    main()
