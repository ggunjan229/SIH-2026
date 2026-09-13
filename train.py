 
from __future__ import annotations
 
import json
from datetime import datetime, timezone
 
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
 
import my_ml_core
 
DATA_PATH = "smart_worker_allocation_20000_v2 (2).csv"
MODEL_PATH = "model.joblib"
REPORT_PATH = "evaluation_report.json"
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5
 
 
def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), my_ml_core.NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), my_ml_core.CATEGORICAL_FEATURES),
        ],
        remainder="drop",  # drops booking_id/worker_id and every excluded column automatically
    )
 
 
def candidate_models() -> dict:
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1),
        "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
    }
 
 
def group_cv_roc_auc(pipeline, X, y, groups, n_splits=CV_FOLDS) -> np.ndarray:
    gkf = GroupKFold(n_splits=n_splits)
    scores = []
    for tr, va in gkf.split(X, y, groups=groups):
        pipeline.fit(X.iloc[tr], y.iloc[tr])
        proba = pipeline.predict_proba(X.iloc[va])[:, 1]
        scores.append(roc_auc_score(y.iloc[va], proba))
    return np.array(scores)
 
 
def main() -> None:
    print(f"[1/6] Loading '{DATA_PATH}' ...")
    df = pd.read_csv(DATA_PATH)
    my_ml_core.validate_schema(df, require_target=True)
    df = my_ml_core.add_relative_features(df, group_col=my_ml_core.GROUP_COLUMN)
    print(f"      {len(df)} rows, {df[my_ml_core.GROUP_COLUMN].nunique()} bookings")
 
    X, y, groups = df[my_ml_core.FEATURE_COLUMNS], df[my_ml_core.TARGET_COLUMN], df[my_ml_core.GROUP_COLUMN]
 
    print(f"[2/6] Group-aware split by booking_id (test_size={TEST_SIZE}) ...")
    splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    assert set(groups.iloc[train_idx]) & set(groups.iloc[test_idx]) == set(), "booking leaked across split!"
    print(f"      train {len(X_train)} rows / test {len(X_test)} rows (held out, untouched until final eval)")
 
    print(f"[3/6] Comparing models via {CV_FOLDS}-fold GroupKFold CV on training split ...")
    cv_results = {}
    for name, model in candidate_models().items():
        pipe = Pipeline([("preprocess", build_preprocessor()), ("model", model)])
        scores = group_cv_roc_auc(pipe, X_train, y_train, groups.iloc[train_idx])
        cv_results[name] = {"mean_roc_auc": float(scores.mean()), "std_roc_auc": float(scores.std())}
        print(f"      {name:26s} mean ROC-AUC = {scores.mean():.4f} (std {scores.std():.4f})")
 
    best_name = max(cv_results, key=lambda k: cv_results[k]["mean_roc_auc"])
    print(f"[4/6] Selected: {best_name} (highest mean CV ROC-AUC, not raw accuracy)")
 
    final_pipeline = Pipeline([("preprocess", build_preprocessor()), ("model", candidate_models()[best_name])])
    final_pipeline.fit(X_train, y_train)
 
    print("[5/6] Evaluating once on held-out test bookings ...")
    y_proba = final_pipeline.predict_proba(X_test)[:, 1]
    y_pred = final_pipeline.predict(X_test)
    cls_metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    for k, v in cls_metrics.items():
        print(f"      {k:12s}: {v}" if k == "confusion_matrix" else f"      {k:12s}: {v:.4f}")
 
    ranking_df = pd.DataFrame({
        my_ml_core.GROUP_COLUMN: df.iloc[test_idx][my_ml_core.GROUP_COLUMN].to_numpy(),
        "score": y_proba,
        "label": y_test.to_numpy(),
    })
    rank_metrics = my_ml_core.evaluate_ranking(ranking_df, "score", "label", my_ml_core.GROUP_COLUMN)
    for k, v in rank_metrics.items():
        print(f"      {k:32s}: {v}")
 
    print(f"[6/6] Saving '{MODEL_PATH}' ...")
    artifact = {
        "pipeline": final_pipeline,
        "model_name": best_name,
        "feature_columns": my_ml_core.FEATURE_COLUMNS,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "cv_model_comparison": cv_results,
        "test_metrics": cls_metrics,
        "ranking_metrics": rank_metrics,
    }
    joblib.dump(artifact, MODEL_PATH)
    with open(REPORT_PATH, "w") as f:
        json.dump({"selected_model": best_name, "cv_model_comparison": cv_results,
                    "test_metrics": cls_metrics, "ranking_metrics": rank_metrics}, f, indent=2)
    print(f"\nDone. Saved: {MODEL_PATH}, {REPORT_PATH}")
 
 
if __name__ == "__main__":
    main()
 