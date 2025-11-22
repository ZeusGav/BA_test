
import argparse
import os
import pandas as pd

def format_hi_small(transactions_path: str,
                    patterns_path: str,
                    output_path: str) -> None:
    """
    Preprocessing for the IBM HI-Small dataset (Kaggle).

    Goal:
    - Mirror the idea of the IBM/Multi-GNN preprocessing step:
      transform raw HI-Small files into a single, ML-ready,
      tabular dataset (formatted_transactions.csv).
    - Focus on a tabular XGBoost baseline and local explainability (SHAP),
      rather than graph models.

    The script performs the following steps:
    1. Load the raw HI-Small transactions CSV.
    2. Parse timestamps and derive simple time-based features.
    3. Create structural flags (same account, same bank, same currency).
    4. Define a binary target label from the 'Is Laundering' column.
    5. Drop pure identifier columns from the feature space.
    6. Apply one-hot encoding to key categorical attributes.
    7. Save the result as a CSV file that is directly usable by ML models.
    """

    # 1) Load transaction data
    df = pd.read_csv(transactions_path)

    if "Is Laundering" not in df.columns:
        raise ValueError("Column 'Is Laundering' not found in transactions file.")

    # 2) Convert timestamp to datetime and derive simple time features
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(
            df["Timestamp"],
            format="%Y/%m/%d %H:%M",
            errors="coerce"
        )
        df["Hour"] = df["Timestamp"].dt.hour
        df["Weekday"] = df["Timestamp"].dt.dayofweek
        df["Weekend"] = (df["Weekday"] >= 5).astype(int)
    else:
        raise ValueError("Column 'Timestamp' not found – please verify the dataset format.")

    # 3) Structural relationship flags (accounts, banks, currencies)
    if {"Account", "Account.1"}.issubset(df.columns):
        df["SameAccount"] = (df["Account"] == df["Account.1"]).astype(int)
    else:
        df["SameAccount"] = 0

    if {"From Bank", "To Bank"}.issubset(df.columns):
        df["SameBank"] = (df["From Bank"] == df["To Bank"]).astype(int)
    else:
        df["SameBank"] = 0

    if {"Receiving Currency", "Payment Currency"}.issubset(df.columns):
        df["SameCurrency"] = (df["Receiving Currency"] == df["Payment Currency"]).astype(int)
    else:
        df["SameCurrency"] = 0

    # 4) Define the target label from 'Is Laundering'
    df["label"] = df["Is Laundering"].astype(int)

    # 5) Remove pure identifier columns from the feature space
    id_cols = [
        "Account",
        "Account.1",
        "From Bank",
        "To Bank",
    ]

    drop_cols = [c for c in id_cols if c in df.columns]
    drop_cols.append("Timestamp")  # we already extracted time-based features

    drop_cols = [c for c in drop_cols if c in df.columns]

    target_col = "label"
    feature_df = df.drop(columns=drop_cols + [target_col, "Is Laundering"])

    # 6) One-hot encode key categorical columns
    categorical_cols = []
    for col in ["Receiving Currency", "Payment Currency", "Payment Format"]:
        if col in feature_df.columns:
            categorical_cols.append(col)

    feature_df = pd.get_dummies(feature_df, columns=categorical_cols, dummy_na=False)

    # Reattach the label as the first column
    out_df = pd.concat(
        [df[target_col].reset_index(drop=True), feature_df.reset_index(drop=True)],
        axis=1
    )

    # 7) Save formatted dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_df.to_csv(output_path, index=False)

    print(f"Formatted dataset saved to: {output_path}")
    print(f"Shape: {out_df.shape}")
    print(f"Positive class share: {out_df['label'].mean():.6f}")


def main():
    parser = argparse.ArgumentParser(
        description="Format IBM HI-Small Kaggle files into an ML-ready CSV."
    )
    parser.add_argument(
        "--transactions_path",
        type=str,
        default="data/raw/HI-Small_Trans.csv"
    )
    parser.add_argument(
        "--patterns_path",
        type=str,
        default="data/raw/HI-Small_Patterns.txt"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/processed/formatted_transactions.csv"
    )
    args = parser.parse_args()

    format_hi_small(
        transactions_path=args.transactions_path,
        patterns_path=args.patterns_path,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
