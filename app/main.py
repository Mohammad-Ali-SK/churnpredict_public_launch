from __future__ import annotations

import gzip
import io
import os
from pathlib import Path
from typing import Annotated, Any

import joblib
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.ml import DatasetError, analyse_labeled_dataset, predict_with_artifact, read_dataset

APP_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(APP_DIR / "models" / "churn_model.joblib")))
FRONTEND_PATH = APP_DIR / "frontend" / "index.html"
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(30 * 1024 * 1024)))
PUBLIC_DEMO = os.getenv("PUBLIC_DEMO", "false").strip().lower() in {"1", "true", "yes"}

app = FastAPI(
    title="ChurnPredict ML API",
    version="1.0.0",
    description="Train and evaluate a churn classifier from labelled data, then score new customer rows.",
)

# Same-origin production deployment is used, but keep local development origins enabled.
origins = [origin.strip() for origin in os.getenv(
    "CORS_ORIGINS",
    "http://localhost:3000,http://localhost:4173,http://localhost:5173,http://127.0.0.1:3000,http://127.0.0.1:5173",
).split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def _load_artifact() -> dict[str, Any] | None:
    if not MODEL_PATH.exists():
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        return None


async def _read_upload(file: UploadFile) -> tuple[bytes, str]:
    filename = file.filename or "upload.csv"
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail=f"File is too large. Maximum upload size is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
    return content, filename


def _dataset_error(exc: DatasetError) -> HTTPException:
    return HTTPException(status_code=422, detail=str(exc))

MAX_DECOMPRESSED_UPLOAD_BYTES = 80 * 1024 * 1024


def _decompress_csv_if_needed(contents: bytes, filename: str) -> tuple[bytes, str]:
    """Expand gzip-compressed CSV uploads with a hard decompressed-size limit."""
    if not filename.lower().endswith(".csv.gz"):
        return contents, filename
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(contents), mode="rb") as stream:
            expanded = stream.read(MAX_DECOMPRESSED_UPLOAD_BYTES + 1)
    except (OSError, EOFError) as exc:
        raise DatasetError("The compressed CSV is invalid or incomplete.") from exc
    if len(expanded) > MAX_DECOMPRESSED_UPLOAD_BYTES:
        raise DatasetError("The compressed CSV exceeds the 80 MiB processing limit.")
    return expanded, filename[:-3]



@app.get("/api/health")
def health() -> dict[str, Any]:
    artifact = _load_artifact()
    return {
        "status": "ok",
        "service": "ChurnPredict ML API",
        "model_available": artifact is not None,
        "model_name": artifact.get("model_name") if artifact else None,
        "api_version": app.version,
    }


@app.get("/api/model")
def model_status() -> dict[str, Any]:
    artifact = _load_artifact()
    if artifact is None:
        return {"trained": False, "message": "No model is stored yet. POST labelled historical data to /api/analyze or /api/train first."}
    return {
        "trained": True,
        "model_name": artifact["model_name"],
        "target_column": artifact["target_column"],
        "positive_class": artifact["positive_class"],
        "trained_rows": artifact["trained_rows"],
        "feature_columns": artifact["feature_columns"],
        "evaluation": artifact["evaluation"],
    }


@app.post("/api/analyze")
async def analyze(
    file: Annotated[UploadFile, File(description="Labelled historical customer data (.csv, .xlsx, .xlsm)")],
    target_column: Annotated[str | None, Form()] = None,
    positive_label: Annotated[str | None, Form()] = None,
) -> dict[str, Any]:
    """Evaluate candidate models using out-of-fold probabilities and save the best fitted model."""
    if PUBLIC_DEMO:
        raise HTTPException(status_code=403, detail="Public demo mode is prediction-only; model training is disabled to protect the shared demo model.")
    contents, filename = await _read_upload(file)
    try:
        df = read_dataset(contents, filename)
        result, artifact = analyse_labeled_dataset(df, target_column or None, positive_label or None, save_model=True)
        assert artifact is not None
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, MODEL_PATH)
        result["dataset"]["name"] = filename
        result["model_saved"] = True
        return result
    except DatasetError as exc:
        raise _dataset_error(exc) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unexpected model-processing error ({type(exc).__name__}). Check backend logs.") from exc


MAX_DYNAMIC_TRAIN_ROWS = 200_000


@app.post("/api/analyze-smart")
async def analyze_smart(
    file: Annotated[
        UploadFile,
        File(description="Labelled customer churn data"),
    ],
    target_column: Annotated[str | None, Form()] = None,
    positive_label: Annotated[str | None, Form()] = None,
) -> dict[str, Any]:
    """Analyse labelled data without replacing the shared production model."""
    contents, filename = await _read_upload(file)

    try:
        contents, filename = _decompress_csv_if_needed(contents, filename)
        df = read_dataset(contents, filename)

        if len(df) > MAX_DYNAMIC_TRAIN_ROWS:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"This demo accepts up to "
                    f"{MAX_DYNAMIC_TRAIN_ROWS:,} rows per request."
                ),
            )

        result, _ = analyse_labeled_dataset(
            df,
            target_column=target_column or None,
            positive_label=positive_label or None,
            save_model=False,
        )

        result["dataset"]["name"] = filename
        result["model_saved"] = False
        result["analysis_mode"] = "dataset_specific"
        result["notes"].append(
            "Results belong to this uploaded dataset. "
            "The shared production model was not replaced."
        )

        return result

    except DatasetError as exc:
        raise _dataset_error(exc) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Unexpected analysis error ({type(exc).__name__}). "
                "Check backend logs."
            ),
        ) from exc

@app.post("/api/train")
async def train(
    file: Annotated[UploadFile, File(description="Labelled historical customer data (.csv, .xlsx, .xlsm)")],
    target_column: Annotated[str | None, Form()] = None,
    positive_label: Annotated[str | None, Form()] = None,
) -> dict[str, Any]:
    """Alias for /api/analyze; returns validation metrics and saves the chosen model."""
    return await analyze(file=file, target_column=target_column, positive_label=positive_label)


@app.post("/api/predict")
async def predict(
    file: Annotated[UploadFile, File(description="New customer rows with columns matching the trained model")],
) -> dict[str, Any]:
    artifact = _load_artifact()
    if artifact is None:
        raise HTTPException(status_code=409, detail="No trained model is available. Upload labelled historical data to /api/train first.")
    contents, filename = await _read_upload(file)
    try:
        df = read_dataset(contents, filename)
        df = df.drop(columns=[artifact["target_column"]], errors="ignore")
        result = predict_with_artifact(df, artifact)
        result["dataset_name"] = filename
        return result
    except DatasetError as exc:
        raise _dataset_error(exc) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unexpected prediction error ({type(exc).__name__}). Check backend logs.") from exc


# The compiled frontend uses client-side routes such as /sample-results. Serve the
# same HTML for those browser routes while leaving all /api/* endpoints untouched.
@app.get("/{full_path:path}", include_in_schema=False)
def frontend_routes(full_path: str) -> FileResponse:
    if full_path.startswith("api/"):
        raise HTTPException(status_code=404, detail="Not Found")
    if not FRONTEND_PATH.is_file():
        raise HTTPException(status_code=404, detail="Frontend file not found")
    return FileResponse(FRONTEND_PATH, media_type="text/html")
