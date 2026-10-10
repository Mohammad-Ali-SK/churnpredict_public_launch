from __future__ import annotations

import io
import re
from dataclasses import dataclass
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
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MAX_ROWS = 100_000
MAX_COLUMNS = 200
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
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=2)),
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
    y_series = pd.Series(y, index=X.index)
    folds = min(5, minority_count)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    candidate_names = ["Logistic Regression", "Random Forest"]
    evaluated: list[dict[str, Any]] = []
    probabilities_by_model: dict[str, np.ndarray] = {}

    for model_name in candidate_names:
        pipeline = build_pipeline(numeric_cols, categorical_cols, model_name)
        try:
            probabilities = cross_val_predict(pipeline, X, y_series, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
            metrics = _metrics(y, probabilities)
            probabilities_by_model[model_name] = probabilities
            evaluated.append({"model": model_name, **metrics})
        except Exception as exc:
            # One candidate failing should not prevent the other model from running.
            evaluated.append({"model": model_name, "error": f"Model could not be evaluated ({type(exc).__name__})."})

    valid = [item for item in evaluated if "roc_auc" in item]
    if not valid:
        raise DatasetError("The model could not be fitted to this dataset. Check for malformed or highly sparse columns.")
    best = max(valid, key=lambda item: (item["roc_auc"], item["f1"]))
    selected_model = best["model"]
    oof_probabilities = probabilities_by_model[selected_model]

    final_pipeline = build_pipeline(numeric_cols, categorical_cols, selected_model)
    final_pipeline.fit(X, y)
    feature_importance = _feature_importance(final_pipeline, numeric_cols, categorical_cols)

    # Use a stable identifier-like column for display when available, but never
    # return the full uploaded customer records to the browser.
    id_candidates = [c for c in cleaned.columns if _norm(c) in {"customerid", "customer_id", "accountid", "userid", "subscriberid", "id"}]
    id_col = id_candidates[0] if id_candidates else None
    predictions: list[dict[str, Any]] = []
    row_limit = min(len(cleaned), MAX_RETURNED_PREDICTIONS)
    ids = cleaned[id_col].tolist() if id_col else cleaned.index.tolist()
    actual = y.tolist()
    for idx in range(row_limit):
        p = float(oof_probabilities[idx])
        predictions.append({
            "row_number": idx + 1,
            "customer_id": _safe_value(ids[idx]),
            "churn_probability": round(p, 6),
            "predicted_churn": bool(p >= 0.5),
            "predicted_label": "Churn" if p >= 0.5 else "Retained",
            "risk_level": _risk_level(p),
            "actual_churn": int(actual[idx]),
        })

    missing_cells = int(cleaned.isna().sum().sum())
    duplicate_rows = int(cleaned.duplicated().sum())
    churn_count = int(y.sum())
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
            "dropped_feature_columns": [],
            "feature_columns": X.columns.tolist(),
            "dropped_feature_columns": [c for c in cleaned.columns if c != target and c not in X.columns],
        },
        "evaluation": _metrics(y, oof_probabilities),
        "cross_validation": {"method": "StratifiedKFold", "folds": folds, "probabilities": "out-of-fold"},
        "model_comparison": evaluated,
        "selected_model": selected_model,
        "feature_importance": feature_importance,
        "predictions": predictions,
        "predictions_truncated": len(cleaned) > row_limit,
        "prediction_count_returned": row_limit,
        "notes": [
            "Evaluation metrics use out-of-fold predictions; each scored row was held out from the model that produced its probability.",
            "Feature importance indicates predictive association, not causation.",
            "Positive class inference should be checked before relying on the churn labels.",
        ],
    }
    artifact = {
        "pipeline": final_pipeline,
        "model_name": selected_model,
        "target_column": target,
        "positive_class": str(positive_class),
        "negative_class": str(next(v for v in pd.unique(cleaned[target]) if str(v) != str(positive_class))),
        "feature_columns": X.columns.tolist(),
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "trained_rows": int(len(cleaned)),
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
