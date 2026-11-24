"""Training, evaluation, persistence, and inference for FinGuard AI."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from etl_and_features import (
    CATEGORICAL_COLUMNS, FEATURE_COLUMNS, NUMERIC_COLUMNS,
    clean_transactions, engineer_features,
)

LOGGER = logging.getLogger(__name__)
ARTIFACT_PATH = Path("model_artifacts.joblib")


def _make_pipeline(classifier: Any) -> Pipeline:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])
    transformer = ColumnTransformer([
        ("numeric", numeric, NUMERIC_COLUMNS),
        ("categorical", categorical, CATEGORICAL_COLUMNS),
    ])
    return Pipeline([("preprocessor", transformer), ("classifier", classifier)])


def _metrics(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    probability = model.predict_proba(x_test)[:, 1]
    predictions = (probability >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_test, probability)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }


def train_and_save(
    raw: pd.DataFrame,
    artifact_path: str | Path = ARTIFACT_PATH,
    random_state: int = 42,
) -> dict[str, Any]:
    """Clean, train, compare, and persist the best model and audit metadata."""
    clean, etl_report = clean_transactions(raw)
    if "is_fraud" not in clean or clean["is_fraud"].nunique() < 2:
        raise ValueError("Training data must include both is_fraud classes (0 and 1).")

    baseline_amount = float(clean["amount"].median())
    featured = engineer_features(clean, baseline_amount)
    x = featured[FEATURE_COLUMNS]
    y = clean["is_fraud"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=random_state, stratify=y
    )

    models = {
        "Random Forest": _make_pipeline(RandomForestClassifier(
            n_estimators=250, min_samples_leaf=2, class_weight="balanced",
            n_jobs=-1, random_state=random_state,
        )),
        # GradientBoostingClassifier has no class_weight argument; equivalent
        # inverse-frequency sample weights are applied during fit.
        "Gradient Boosting": _make_pipeline(GradientBoostingClassifier(
            n_estimators=120, learning_rate=0.06, max_depth=2,
            random_state=random_state,
        )),
    }
    results: dict[str, dict[str, Any]] = {}
    fitted: dict[str, Pipeline] = {}
    class_counts = y_train.value_counts()
    weights = y_train.map({
        label: len(y_train) / (len(class_counts) * count)
        for label, count in class_counts.items()
    }).to_numpy()

    for name, pipeline in models.items():
        if name == "Gradient Boosting":
            pipeline.fit(x_train, y_train, classifier__sample_weight=weights)
        else:
            pipeline.fit(x_train, y_train)
        fitted[name] = pipeline
        results[name] = _metrics(pipeline, x_test, y_test)

    best_name = max(results, key=lambda name: results[name]["roc_auc"])
    best_model = fitted[best_name]
    transformer = best_model.named_steps["preprocessor"]
    feature_names = transformer.get_feature_names_out()
    importances = best_model.named_steps["classifier"].feature_importances_
    importance_rows = sorted(
        [
            {"feature": str(name).replace("numeric__", "").replace("categorical__", ""),
             "importance": float(value)}
            for name, value in zip(feature_names, importances)
        ],
        key=lambda item: item["importance"],
        reverse=True,
    )

    artifacts = {
        "model": best_model,
        "model_name": best_name,
        "metrics": results[best_name],
        "all_model_metrics": results,
        "feature_importances": importance_rows,
        "feature_columns": FEATURE_COLUMNS,
        "baseline_amount": baseline_amount,
        "etl_report": etl_report,
        "threshold": 0.5,
    }
    output_path = Path(artifact_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifacts, output_path)
    LOGGER.info("Saved %s model artifacts to %s", best_name, output_path)
    return artifacts


def load_artifacts(path: str | Path = ARTIFACT_PATH) -> dict[str, Any]:
    return joblib.load(path)


def score_transaction(
    transaction: dict[str, Any] | pd.Series,
    artifacts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return fraud probability, risk tier, and model-informed risk indicators."""
    artifacts = artifacts or load_artifacts()
    raw = pd.DataFrame([dict(transaction)])
    clean, _ = clean_transactions(raw)
    featured = engineer_features(clean, artifacts["baseline_amount"])
    x = featured[artifacts["feature_columns"]]
    model = artifacts["model"]
    probability = float(model.predict_proba(x)[0, 1])

    tier = "Low" if probability < 0.25 else "Medium" if probability < 0.65 else "Critical"

    # Rank active encoded features by the fitted tree model's global importance.
    # This is an importance-based indicator, not a causal or SHAP explanation.
    transformed = model.named_steps["preprocessor"].transform(x)
    names = model.named_steps["preprocessor"].get_feature_names_out()
    importances = model.named_steps["classifier"].feature_importances_
    row = transformed.toarray()[0] if hasattr(transformed, "toarray") else np.asarray(transformed)[0]
    active = np.flatnonzero(np.abs(row) > 1e-10)
    ranked = sorted(active, key=lambda idx: importances[idx], reverse=True)
    reasons = []
    for idx in ranked:
        if importances[idx] <= 0:
            continue
        name = str(names[idx]).replace("numeric__", "").replace("categorical__", "")
        value = row[idx]
        reasons.append(f"{name.replace('_', ' ').title()} (importance {importances[idx]:.3f})")
        if len(reasons) == 3:
            break

    return {
        "fraud_probability": probability,
        "risk_tier": tier,
        "top_risk_factors": reasons or ["No dominant model indicators identified"],
        "model_name": artifacts["model_name"],
    }


def score_batch(
    uploaded: pd.DataFrame,
    artifacts: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Clean and score a CSV-compatible batch; original columns are retained."""
    artifacts = artifacts or load_artifacts()
    clean, _ = clean_transactions(uploaded)
    featured = engineer_features(clean, artifacts["baseline_amount"])
    probabilities = artifacts["model"].predict_proba(
        featured[artifacts["feature_columns"]]
    )[:, 1]
    output = clean.copy()
    output["fraud_probability"] = probabilities
    output["risk_tier"] = pd.cut(
        probabilities, bins=[-0.001, 0.25, 0.65, 1.0],
        labels=["Low", "Medium", "Critical"], include_lowest=True,
    ).astype(str)
    return output