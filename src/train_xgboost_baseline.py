
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


def main():
    # 1) Load configuration for AML dataset path and target column
    with open("data_config.json", "r") as f:
        cfg = json.load(f)

    data_path = cfg.get("aml_data", "data/processed/formatted_transactions.csv")
    target_column = cfg.get("target_column", "label")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Formatted dataset not found at: {data_path}")

    # 2) Load data and split into features and target
    df = pd.read_csv(data_path)

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")

    drop_cols = [target_column]
    if "transaction_id" in df.columns:
        drop_cols.append("transaction_id")

    X = df.drop(columns=drop_cols)
    y = df[target_column].astype(int)

    # 3) Train/test split (stratified to respect class imbalance)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # 4) Handle class imbalance via scale_pos_weight
    pos = y_train.sum()
    neg = len(y_train) - pos
    scale_pos_weight = neg / pos if pos > 0 else 1.0

    # 5) Define a clear, interpretable XGBoost baseline model
    model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
        scale_pos_weight=scale_pos_weight,
    )

    model.fit(X_train, y_train)

    # 6) Evaluate using ROC-AUC and PR-AUC (average precision)
    proba_test = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, proba_test)
    pr_auc = average_precision_score(y_test, proba_test)

    print(f"Test ROC-AUC: {roc_auc:.4f}")
    print(f"Test PR-AUC : {pr_auc:.4f}")
    print(f"Positive share train: {y_train.mean():.6f}, test: {y_test.mean():.6f}")

    # 7) Save model and metrics for later SHAP and LLM-based explanations
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "xgb_hi_small_baseline.pkl")
    metrics_path = os.path.join("models", "xgb_hi_small_baseline_metrics.json")

    joblib.dump(model, model_path)

    with open(metrics_path, "w") as f:
        json.dump({"roc_auc": roc_auc, "pr_auc": pr_auc}, f, indent=2)

    print(f"Model saved to: {model_path}")
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
