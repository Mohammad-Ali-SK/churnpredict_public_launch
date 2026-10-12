from __future__ import annotations

import io
import re
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MAX_ROWS = 200_000
MAX_COLUMNS = 100
MAX_RETURNED_PREDICTIONS = 5_000

TARGET_CANDIDATES = {
    "churn", "churned", "attrition", "attrited", "attritionflag",
    "exited", "left", "cancelled", "canceled", "iscustomerchurn",
    "customerchurn", "status",
}
POSITIVE_LABELS = {
    "yes", "true", "1", "churn", "churned", "left", "exited",
    "attrition", "attrited", "attritedcustomer", "cancelled", "canceled",
    "terminated", "inactive", "lost", "departed",
}
NEGATIVE_LABELS = {
    "no", "false", "0", "retained", "stay", "stayed", "active",
    "existingcustomer", "notchurned", "nochurn", "notleft", "notexited",
}


class DatasetError(ValueError):
    """Raised when an uploaded dataset cannot be analysed safely."""


def _norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def read_dataset(contents: bytes, filename: str) -> pd.DataFrame:
    """Load CSV or Excel bytes and standardise common blank-value representations."""
    suffix = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    try:
        if suffix == "csv":
            # UTF-8 first; common Windows CSV exports may use cp1252.
            try:
                df = pd.read_csv(io.BytesIO(contents), low_memory=False)
            except UnicodeDecodeError:
                df = pd.read_csv(io.BytesIO(contents), encoding="cp1252", low_memory=False)
        elif suffix in {"xlsx", "xlsm"}:
            df = pd.read_excel(io.BytesIO(contents), engine="openpyxl")
        else:
            raise DatasetError("Unsupported file type. Upload a .csv, .xlsx, or .xlsm file.")
    except DatasetError:
        raise
    except Exception as exc:
        raise DatasetError(f"Could not read this file. Check that it is a valid CSV or Excel file. ({type(exc).__name__})") from exc

    if df.empty:
        raise DatasetError("The uploaded file has no data rows.")
    df.columns = [str(c).strip() for c in df.columns]
    if any(not c for c in df.columns):
        df.columns = [c if c else f"unnamed_column_{i + 1}" for i, c in enumerate(df.columns)]
    if df.columns.duplicated().any():
        duplicates = df.columns[df.columns.duplicated()].tolist()
        raise DatasetError(f"Column names must be unique. Duplicate columns: {duplicates[:5]}")

    # Blank strings are missing values for imputation and data-quality metrics.
    df = df.replace(r"^\s*$", np.nan, regex=True)
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if df.empty:
        raise DatasetError("The uploaded file contains only blank rows or columns.")
    if len(df) > MAX_ROWS:
        raise DatasetError(f"This starter backend supports up to {MAX_ROWS:,} data rows per request.")
    if len(df.columns) > MAX_COLUMNS:
        raise DatasetError(f"This starter backend supports up to {MAX_COLUMNS} columns per request.")

    # Convert object columns that are almost entirely numeric (e.g. charge columns
    # exported with a few blanks) without trying to convert ID/category fields.
    for col in df.select_dtypes(include=["object"]).columns:
        parsed = pd.to_numeric(df[col], errors="coerce")
        non_null = int(df[col].notna().sum())
        if non_null and parsed.notna().sum() / non_null >= 0.90:
            df[col] = parsed
    return df


def detect_target_column(df: pd.DataFrame) -> str | None:
    by_normalized = {_norm(c): c for c in df.columns}
    for candidate in TARGET_CANDIDATES:
        if candidate in by_normalized:
            return by_normalized[candidate]
    # More specific fallback names; avoid selecting arbitrary "status" columns
    # unless the exact target-name match above found one.
    for normalized, original in by_normalized.items():
        if normalized in {"churnflag", "attritionflag", "churnstatus", "customerexited"}:
            return original
    return None


def resolve_target(df: pd.DataFrame, target_column: str | None) -> str:
    if target_column:
        if target_column not in df.columns:
            raise DatasetError(
                f"Target column '{target_column}' was not found. Available columns: {', '.join(map(str, df.columns[:30]))}"
            )
        return target_column
    detected = detect_target_column(df)
    if detected is None:
        raise DatasetError(
            "No churn target column was detected. To evaluate/train a model, upload historical data containing a label such as 'Churn' or 'Attrition', and select that target column. For unlabeled customers, train a model first and then use /api/predict."
        )
    return detected


def infer_positive_class(values: pd.Series, positive_label: str | None = None) -> tuple[Any, str]:
    classes = list(pd.unique(values.dropna()))
    if len(classes) != 2:
        raise DatasetError(f"The target must have exactly 2 classes for binary churn prediction; found {len(classes)}.")

    if positive_label is not None and str(positive_label).strip():
        requested = str(positive_label).strip().lower()
        matches = [v for v in classes if str(v).strip().lower() == requested]
        if not matches:
            raise DatasetError(f"Positive label '{positive_label}' is not present in the target. Available labels: {[str(v) for v in classes]}")
        return matches[0], "user_selected"

    for value in classes:
        if _norm(value) in POSITIVE_LABELS:
            return value, "label_name"
    # For numeric / unusual binary labels, prefer 1 if it exists.
    for value in classes:
        if str(value).strip() == "1":
            return value, "numeric_1"
    # Fallback: designate the minority class as churn. Explicitly report that
    # this is an inference so callers can verify it before using results.
    counts = values.value_counts(dropna=True)
    smallest_count = counts.min()
    minority = [v for v in classes if counts[v] == smallest_count]
    if len(minority) == 1:
        return minority[0], "minority_class_fallback"
    raise DatasetError(
        f"Could not reliably infer which class means churn because the classes are balanced. Supply positive_label. Available labels: {[str(v) for v in classes]}"
    )


def _looks_like_identifier(column: str, values: pd.Series, row_count: int) -> bool:
    name = _norm(column)
    explicit_id = (
        name in {"id", "customerid", "accountid", "userid", "subscriberid", "rowid", "recordid"}
        or name.endswith("identifier")
        or name.endswith("_id")
    )
    unique_ratio = values.nunique(dropna=True) / max(row_count, 1)
    non_numeric = not pd.api.types.is_numeric_dtype(values)
    return explicit_id or (non_numeric and unique_ratio >= 0.98 and values.nunique(dropna=True) >= 20)


def prepare_features(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, list[str], list[str]]:
    X = df.drop(columns=[target]).copy()
    dropped: list[str] = []
    for col in list(X.columns):
        series = X[col]
        if series.isna().all() or series.nunique(dropna=True) <= 1 or _looks_like_identifier(str(col), series, len(df)):
            dropped.append(str(col))
            X = X.drop(columns=[col])

    if X.shape[1] == 0:
        raise DatasetError("No usable feature columns remain after excluding the target and constant/ID-like columns.")
    # Ensure simple scalar columns only; duplicate names were checked on import.
    X = X.replace([np.inf, -np.inf], np.nan)
    numeric_cols = X.select_dtypes(include=[np.number, "bool"]).columns.astype(str).tolist()
    categorical_cols = [str(c) for c in X.columns if str(c) not in numeric_cols]
    # Preserve column order and names as strings.
    X.columns = [str(c) for c in X.columns]
    return X, numeric_cols, categorical_cols


def build_pipeline(numeric_cols: list[str], categorical_cols: list[str], model_name: str) -> Pipeline:
    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_cols:
        numeric_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", numeric_pipe, numeric_cols))
    if categorical_cols:
        categorical_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=2, max_categories=50)),
        ])
        transformers.append(("cat", categorical_pipe, categorical_cols))
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    if model_name == "Logistic Regression":
        classifier = LogisticRegression(max_iter=1500, class_weight="balanced", random_state=42)
    elif model_name == "Random Forest":
        classifier = RandomForestClassifier(
            n_estimators=160,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
    else:
        raise DatasetError(f"Unsupported model: {model_name}")
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def _metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    return {
        "accuracy": round(float(accuracy_score(y_true, predictions)), 4),
        "precision": round(float(precision_score(y_true, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, predictions, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, probabilities)), 4),
        "pr_auc": round(float(average_precision_score(y_true, probabilities)), 4),
        "threshold": threshold,
        "confusion_matrix": {
            "labels": ["Retained", "Churn"],
            "true_negative": int(matrix[0, 0]),
            "false_positive": int(matrix[0, 1]),
            "false_negative": int(matrix[1, 0]),
            "true_positive": int(matrix[1, 1]),
        },
    }


def _feature_importance(pipeline: Pipeline, numeric_cols: list[str], categorical_cols: list[str]) -> list[dict[str, Any]]:
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    try:
        transformed_names = preprocessor.get_feature_names_out()
    except Exception:
        return []

    if hasattr(classifier, "feature_importances_"):
        values = np.asarray(classifier.feature_importances_, dtype=float)
    elif hasattr(classifier, "coef_"):
        values = np.mean(np.abs(np.asarray(classifier.coef_, dtype=float)), axis=0)
    else:
        return []
    if len(values) != len(transformed_names):
        return []

    original_names = numeric_cols + categorical_cols
    totals: dict[str, float] = {name: 0.0 for name in original_names}
    for transformed_name, value in zip(transformed_names, values):
        if "__" in transformed_name:
            feature_name = transformed_name.split("__", 1)[1]
        else:
            feature_name = transformed_name
        matches = [name for name in original_names if feature_name == name or feature_name.startswith(name + "_")]
        if matches:
            totals[max(matches, key=len)] += float(value)

    denom = sum(totals.values()) or 1.0
    ranked = sorted(totals.items(), key=lambda item: item[1], reverse=True)
    return [
        {"feature": name, "importance": round(value / denom, 6), "importance_percent": round(100 * value / denom, 2)}
        for name, value in ranked[:15]
    ]


def _risk_level(probability: float) -> str:
    if probability >= 0.70:
        return "High"
    if probability >= 0.30:
        return "Medium"
    return "Low"


def _safe_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return str(value) if not isinstance(value, (str, int, float, bool)) else value


def analyse_labeled_dataset(
    df: pd.DataFrame,
    target_column: str | None = None,
    positive_label: str | None = None,
    save_model: bool = False,
    force_holdout: bool = False,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    target = resolve_target(df, target_column)
    cleaned = df.loc[df[target].notna()].copy()
    if len(cleaned) < 10:
        raise DatasetError("At least 10 rows with a non-empty target label are required to evaluate a model.")

    positive_class, inference_method = infer_positive_class(cleaned[target], positive_label)
    y = (cleaned[target].astype(str) == str(positive_class)).astype(int).to_numpy()
    counts = np.bincount(y, minlength=2)
    minority_count = int(counts.min())
    if minority_count < 2:
        raise DatasetError("The target needs at least 2 rows in each class. More labeled churn and retained examples are required.")

    X, numeric_cols, categorical_cols = prepare_features(cleaned, target)
    large_mode = force_holdout or len(cleaned) > 10_000

    if large_mode:
        if minority_count < 5:
            raise DatasetError("For datasets over 10,000 rows, at least 5 rows in each target class are required for holdout evaluation.")
        train_indices, prediction_indices = train_test_split(
            np.arange(len(X)),
            test_size=0.20,
            random_state=42,
            stratify=y,
        )
        prediction_indices = np.sort(prediction_indices)
        candidate_names = ["Logistic Regression"]
        cv = None
        folds = 1
    else:
        prediction_indices = np.arange(len(X))
        train_indices = None
        folds = min(5, minority_count)
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        candidate_names = ["Logistic Regression", "Random Forest"]

    evaluated: list[dict[str, Any]] = []
    probabilities_by_model: dict[str, np.ndarray] = {}
    pipelines_by_model: dict[str, Pipeline] = {}

    for model_name in candidate_names:
        pipeline = build_pipeline(numeric_cols, categorical_cols, model_name)
        try:
            if large_mode:
                assert train_indices is not None
                pipeline.fit(X.iloc[train_indices], y[train_indices])
                probabilities = pipeline.predict_proba(X.iloc[prediction_indices])[:, 1]
                metrics = _metrics(y[prediction_indices], probabilities)
            else:
                assert cv is not None
                probabilities = cross_val_predict(
                    pipeline, X, y, cv=cv, method="predict_proba", n_jobs=1
                )[:, 1]
                metrics = _metrics(y, probabilities)
                pipeline.fit(X, y)

            probabilities_by_model[model_name] = probabilities
            pipelines_by_model[model_name] = pipeline
            evaluated.append({"model": model_name, **metrics})
        except Exception as exc:
            evaluated.append({
                "model": model_name,
                "error": f"Model could not be evaluated ({type(exc).__name__}).",
            })

    valid = [item for item in evaluated if "roc_auc" in item]
    if not valid:
        raise DatasetError("The model could not be fitted to this dataset. Check for malformed or highly sparse columns.")

    best = max(valid, key=lambda item: (item["roc_auc"], item["f1"]))
    selected_model = best["model"]
    evaluation_probabilities = probabilities_by_model[selected_model]
    evaluation = {key: value for key, value in best.items() if key != "model"}

    if large_mode and not save_model:
        # For dataset-specific analysis, reuse the validation-trained model.
        final_pipeline = pipelines_by_model[selected_model]
    else:
        # Small-data artifact training, or an explicitly requested saved model.
        final_pipeline = build_pipeline(numeric_cols, categorical_cols, selected_model)
        final_pipeline.fit(X, y)

    feature_importance = _feature_importance(final_pipeline, numeric_cols, categorical_cols)

    id_candidates = [
        c for c in cleaned.columns
        if _norm(c) in {"customerid", "customer_id", "accountid", "userid", "subscriberid", "id"}
    ]
    id_col = id_candidates[0] if id_candidates else None
    ids = cleaned[id_col].tolist() if id_col else cleaned.index.tolist()

    predictions: list[dict[str, Any]] = []
    row_limit = min(len(prediction_indices), MAX_RETURNED_PREDICTIONS)
    for position in range(row_limit):
        idx = int(prediction_indices[position])
        probability = float(evaluation_probabilities[position])
        predictions.append({
            "row_number": idx + 1,
            "customer_id": _safe_value(ids[idx]),
            "churn_probability": round(probability, 6),
            "predicted_churn": bool(probability >= 0.5),
            "predicted_label": "Churn" if probability >= 0.5 else "Retained",
            "risk_level": _risk_level(probability),
            "actual_churn": int(y[idx]),
        })

    missing_cells = int(cleaned.isna().sum().sum())
    duplicate_rows = int(cleaned.duplicated().sum())
    churn_count = int(y.sum())

    if large_mode:
        evaluation_description = (
            "Evaluation metrics use a stratified 20% held-out validation set. "
            "Returned predictions are for held-out rows only."
        )
        validation_info = {
            "method": "Stratified train/test holdout",
            "folds": 1,
            "probabilities": "held-out validation",
        }
    else:
        evaluation_description = (
            "Evaluation metrics use out-of-fold predictions; each scored row was "
            "held out from the model that produced its probability."
        )
        validation_info = {
            "method": "StratifiedKFold",
            "folds": folds,
            "probabilities": "out-of-fold",
        }

    result = {
        "status": "ok",
        "dataset": {
            "name": "uploaded_dataset",
            "rows": int(len(cleaned)),
            "columns": int(cleaned.shape[1]),
            "target_column": target,
            "target_classes": [str(v) for v in pd.unique(cleaned[target])],
            "positive_class": str(positive_class),
            "positive_class_inference": inference_method,
            "missing_cells": missing_cells,
            "duplicate_rows": duplicate_rows,
            "churn_count": churn_count,
            "retained_count": int(len(y) - churn_count),
            "churn_rate": round(100 * churn_count / len(y), 2),
            "feature_columns": X.columns.tolist(),
            "dropped_feature_columns": [
                c for c in cleaned.columns if c != target and c not in X.columns
            ],
        },
        "evaluation": evaluation,
        "cross_validation": validation_info,
        "model_comparison": evaluated,
        "selected_model": selected_model,
        "feature_importance": feature_importance,
        "predictions": predictions,
        "predictions_truncated": len(prediction_indices) > row_limit,
        "prediction_count_returned": row_limit,
        "notes": [
            evaluation_description,
            "Feature importance indicates predictive association, not causation.",
            "Positive class inference should be checked before relying on the churn labels.",
        ],
    }

    negative_classes = [v for v in pd.unique(cleaned[target]) if str(v) != str(positive_class)]
    if not negative_classes:
        raise DatasetError("A negative target class could not be identified.")

    artifact = {
        "pipeline": final_pipeline,
        "model_name": selected_model,
        "target_column": target,
        "positive_class": str(positive_class),
        "negative_class": str(negative_classes[0]),
        "feature_columns": X.columns.tolist(),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "trained_rows": int(len(cleaned) if save_model else (
            len(train_indices) if large_mode and train_indices is not None else len(cleaned)
        )),
        "evaluation": result["evaluation"],
        "feature_importance": feature_importance,
    }
    return result, artifact if save_model else None


def predict_with_artifact(df: pd.DataFrame, artifact: dict[str, Any]) -> dict[str, Any]:
    required = list(artifact["feature_columns"])
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DatasetError(f"Prediction file is missing {len(missing)} required feature column(s): {', '.join(missing[:20])}")
    X = df[required].copy().replace([np.inf, -np.inf], np.nan)
    pipeline: Pipeline = artifact["pipeline"]
    probabilities = pipeline.predict_proba(X)[:, 1]
    id_candidates = [c for c in df.columns if _norm(c) in {"customerid", "customer_id", "accountid", "userid", "subscriberid", "id"}]
    id_col = id_candidates[0] if id_candidates else None
    ids = df[id_col].tolist() if id_col else df.index.tolist()
    limit = min(len(df), MAX_RETURNED_PREDICTIONS)
    predictions = []
    for idx in range(limit):
        p = float(probabilities[idx])
        predictions.append({
            "row_number": idx + 1,
            "customer_id": _safe_value(ids[idx]),
            "churn_probability": round(p, 6),
            "predicted_churn": bool(p >= 0.5),
            "predicted_label": "Churn" if p >= 0.5 else "Retained",
            "risk_level": _risk_level(p),
        })
    return {
        "status": "ok",
        "model": {
            "name": artifact["model_name"],
            "target_column": artifact["target_column"],
            "positive_class": artifact["positive_class"],
            "trained_rows": artifact["trained_rows"],
            "evaluation": artifact["evaluation"],
        },
        "dataset": {"rows": int(len(df)), "columns": int(len(df.columns))},
        "predictions": predictions,
        "predictions_truncated": len(df) > limit,
        "prediction_count_returned": limit,
        "notes": ["Scores come from the trained model; predictive quality depends on the training data and evaluation metrics."],
    }

# Free-plan CSV processing safeguards. Full-file profiling and scoring are chunked;
# only a bounded stratified sample is retained for model fitting.
MAX_STREAM_SAMPLE_ROWS = 5_000
MAX_STREAM_CHUNK_ROWS = 2_000
MAX_STREAM_PREDICTIONS = 1_000


def analyse_csv_file(
    path: str | Path,
    target_column: str | None = None,
    positive_label: str | None = None,
) -> dict[str, Any]:
    """Analyse a CSV in bounded chunks, train on a stratified sample, and score all labelled rows.

    Evaluation metrics are calculated on a holdout from the training sample. Full-dataset
    summary counts and risk-band counts are scanned separately and are not training metrics.
    """
    csv_path = Path(path)
    try:
        header_frame = pd.read_csv(csv_path, nrows=0, dtype="string")
    except Exception as exc:
        raise DatasetError(f"Could not read CSV header ({type(exc).__name__}).") from exc

    columns = [str(c).strip() or f"unnamed_column_{i + 1}" for i, c in enumerate(header_frame.columns)]
    if not columns:
        raise DatasetError("The CSV has no columns.")
    if len(columns) > MAX_COLUMNS:
        raise DatasetError(f"This backend supports up to {MAX_COLUMNS} columns per request.")
    if len(set(columns)) != len(columns):
        raise DatasetError("Column names must be unique after trimming whitespace.")

    skeleton = pd.DataFrame(columns=columns)
    target = resolve_target(skeleton, target_column or None)
    target_index = columns.index(target)
    random_gen = random.Random(42)

    non_null_any = {c: 0 for c in columns}
    missing_by_col = {c: 0 for c in columns}
    class_counts: dict[str, int] = {}
    class_order: list[str] = []
    class_reservoirs: dict[str, pd.DataFrame] = {}
    class_seen: dict[str, int] = {}
    seen_hashes: set[int] = set()
    duplicate_rows = 0
    total_nonblank_rows = 0
    labelled_rows = 0
    priority_col = "__churnpredict_sample_priority__"
    while priority_col in columns:
        priority_col += "_"

    try:
        reader = pd.read_csv(csv_path, chunksize=MAX_STREAM_CHUNK_ROWS, dtype="string")
        for chunk in reader:
            if len(chunk.columns) != len(columns):
                raise DatasetError("CSV column count changed while reading the file.")
            chunk.columns = columns
            chunk = chunk.replace(r"^\s*$", pd.NA, regex=True)
            total_nonblank_rows += int(chunk.notna().any(axis=1).sum())
            if total_nonblank_rows > MAX_ROWS:
                raise DatasetError(f"This backend supports up to {MAX_ROWS:,} data rows per request.")

            non_null_counts = chunk.notna().sum()
            for col, count in non_null_counts.items():
                non_null_any[col] += int(count)

            labelled = chunk.loc[chunk[target].notna()]
            if labelled.empty:
                continue
            labelled_rows += len(labelled)
            for col, count in labelled.isna().sum().items():
                missing_by_col[col] += int(count)

            hashes = pd.util.hash_pandas_object(labelled, index=False).to_numpy()
            for value in hashes:
                hv = int(value)
                if hv in seen_hashes:
                    duplicate_rows += 1
                else:
                    seen_hashes.add(hv)

            labels = labelled[target].astype(str)
            value_counts = labels.value_counts(dropna=False)
            for label, count in value_counts.items():
                label = str(label)
                if label not in class_counts:
                    class_order.append(label)
                    class_counts[label] = 0
                    class_seen[label] = 0
                    class_reservoirs[label] = pd.DataFrame(columns=columns + [priority_col])
                class_counts[label] += int(count)
                if len(class_order) > 2:
                    raise DatasetError("Large-file analysis on the Free plan requires a binary churn target (two classes).")

            # Vectorised random priorities let us keep at most 5,000 candidate rows per class.
            priorities = np.asarray([random_gen.random() for _ in range(len(labelled))], dtype=float)
            label_values = labels.to_numpy(dtype=str)
            for label in labels.unique().tolist():
                mask = label_values == str(label)
                subset = labelled.loc[mask].copy()
                subset[priority_col] = priorities[mask]
                class_seen[str(label)] += len(subset)
                existing = class_reservoirs[str(label)]
                combined = (
                    subset.copy()
                    if existing.empty
                    else pd.concat([existing, subset], ignore_index=True)
                )
                if len(combined) > MAX_STREAM_SAMPLE_ROWS:
                    combined = combined.nsmallest(MAX_STREAM_SAMPLE_ROWS, priority_col)
                class_reservoirs[str(label)] = combined
    except DatasetError:
        raise
    except Exception as exc:
        raise DatasetError(f"Could not process CSV in chunks ({type(exc).__name__}).") from exc

    if total_nonblank_rows == 0:
        raise DatasetError("The uploaded CSV contains no data rows.")
    if labelled_rows < 10:
        raise DatasetError("At least 10 rows with a non-empty target label are required to evaluate a model.")
    if len(class_order) != 2:
        raise DatasetError("Large-file analysis requires exactly two target classes, such as Yes/No or 1/0.")

    positive_class, inference_method = infer_positive_class(pd.Series(class_order, dtype="string"), positive_label)
    positive_key = str(positive_class)
    if positive_key not in class_counts:
        raise DatasetError("The selected positive label does not match a target value in the CSV.")
    positive_count = int(class_counts[positive_key])
    negative_count = int(labelled_rows - positive_count)
    if min(positive_count, negative_count) < 5 and total_nonblank_rows > MAX_STREAM_SAMPLE_ROWS:
        raise DatasetError("For large CSVs, at least 5 labelled rows in each target class are required.")

    usable_columns = [c for c in columns if non_null_any[c] > 0]
    full_missing_cells = int(sum(missing_by_col[c] for c in usable_columns))
    target_classes = class_order[:]

    # Small files keep their original full-data cross-validation behavior.
    if total_nonblank_rows <= MAX_STREAM_SAMPLE_ROWS:
        try:
            df = read_dataset(csv_path.read_bytes(), csv_path.name)
        except Exception as exc:
            if isinstance(exc, DatasetError):
                raise
            raise DatasetError(f"Could not read CSV ({type(exc).__name__}).") from exc
        result, _ = analyse_labeled_dataset(df, target, positive_label, save_model=False)
        return result

    sample_size = min(MAX_STREAM_SAMPLE_ROWS, labelled_rows)
    positive_quota = int(round(sample_size * positive_count / labelled_rows))
    positive_quota = max(5, min(sample_size - 5, positive_quota))
    quotas = {positive_key: positive_quota}
    negative_key = next(k for k in class_order if k != positive_key)
    quotas[negative_key] = sample_size - positive_quota

    selected_rows: list[dict[str, Any]] = []
    for label in class_order:
        reservoir = class_reservoirs[label]
        quota = quotas[label]
        if len(reservoir) < quota:
            raise DatasetError("Could not create a sufficiently representative training sample from this CSV.")
        chosen = reservoir.nsmallest(quota, priority_col).drop(columns=[priority_col])
        selected_rows.extend(chosen.to_dict(orient="records"))

    sample_raw = pd.DataFrame(selected_rows, columns=columns)
    sample_bytes = sample_raw.to_csv(index=False).encode("utf-8")
    sample_df = read_dataset(sample_bytes, "churnpredict_training_sample.csv")
    result, artifact = analyse_labeled_dataset(
        sample_df,
        target_column=target,
        positive_label=positive_key,
        save_model=True,
        force_holdout=True,
    )
    if artifact is None:
        raise DatasetError("The sampled model artifact was not created.")

    pipeline: Pipeline = artifact["pipeline"]
    feature_columns = list(artifact["feature_columns"])
    numeric_columns = set(artifact["numeric_columns"])
    id_candidates = [c for c in columns if _norm(c) in {"customerid", "accountid", "userid", "subscriberid", "id", "rowid", "recordid"}]
    id_col = id_candidates[0] if id_candidates else None

    risk_counts = {"high": 0, "medium": 0, "low": 0, "predicted_churn": 0, "total": 0}
    full_predictions: list[dict[str, Any]] = []
    source_offset = 0
    try:
        reader = pd.read_csv(csv_path, chunksize=MAX_STREAM_CHUNK_ROWS, dtype="string")
        for chunk in reader:
            chunk.columns = columns
            chunk = chunk.replace(r"^\s*$", pd.NA, regex=True)
            labelled = chunk.loc[chunk[target].notna()]
            if not labelled.empty:
                X = pd.DataFrame(index=labelled.index)
                for col in feature_columns:
                    if col in numeric_columns:
                        numeric = pd.to_numeric(labelled[col], errors="coerce")
                        X[col] = numeric.to_numpy(dtype=np.float64, na_value=np.nan)
                    else:
                        categorical = labelled[col].astype(object)
                        X[col] = categorical.where(pd.notna(categorical), np.nan)
                X = X.replace([np.inf, -np.inf], np.nan)
                probs = pipeline.predict_proba(X)[:, 1]
                target_values = labelled[target].astype(str).tolist()
                ids = labelled[id_col].tolist() if id_col else list(range(source_offset + 1, source_offset + len(chunk) + 1))
                row_positions = np.flatnonzero(chunk[target].notna().to_numpy())
                for local_idx, probability in enumerate(probs):
                    probability = float(probability)
                    risk = _risk_level(probability)
                    risk_counts[{"High": "high", "Medium": "medium", "Low": "low"}[risk]] += 1
                    risk_counts["predicted_churn"] += int(probability >= 0.5)
                    risk_counts["total"] += 1
                    if len(full_predictions) < MAX_STREAM_PREDICTIONS:
                        original_row_number = source_offset + int(row_positions[local_idx]) + 1
                        customer_id = _safe_value(ids[local_idx])
                        full_predictions.append({
                            "row_number": original_row_number,
                            "customer_id": customer_id,
                            "churn_probability": round(probability, 6),
                            "predicted_churn": bool(probability >= 0.5),
                            "predicted_label": "Churn" if probability >= 0.5 else "Retained",
                            "risk_level": risk,
                            "actual_churn": int(target_values[local_idx] == positive_key),
                        })
            source_offset += len(chunk)
    except Exception as exc:
        if isinstance(exc, DatasetError):
            raise
        raise DatasetError(f"Could not score the full CSV in chunks ({type(exc).__name__}).") from exc

    dataset = result["dataset"]
    dataset.update({
        "name": csv_path.name,
        "rows": int(labelled_rows),
        "columns": int(len(usable_columns)),
        "target_column": target,
        "target_classes": target_classes,
        "positive_class": positive_key,
        "positive_class_inference": inference_method,
        "missing_cells": full_missing_cells,
        "duplicate_rows": int(duplicate_rows),
        "churn_count": positive_count,
        "retained_count": negative_count,
        "churn_rate": round(100 * positive_count / labelled_rows, 2),
    })
    result["predictions"] = full_predictions
    result["predictions_truncated"] = labelled_rows > len(full_predictions)
    result["prediction_count_returned"] = len(full_predictions)
    result["risk_summary"] = risk_counts
    result["cross_validation"]["training_sample_rows"] = int(sample_size)
    result["cross_validation"]["full_dataset_rows"] = int(labelled_rows)
    result["notes"] = list(result.get("notes", [])) + [
        f"The classifier was trained on a stratified sample of {sample_size:,} labelled rows. Evaluation metrics come from the sample's held-out validation split, not from all rows.",
        f"Risk-band counts and the first {len(full_predictions):,} detailed predictions were computed by scoring all {labelled_rows:,} labelled rows in chunks.",
        "Feature importance indicates predictive association, not causation.",
    ]
    return result
