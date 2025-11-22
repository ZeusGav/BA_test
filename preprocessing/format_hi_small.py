import argparse
import os
import pandas as pd

def format_hi_small(transactions_path: str,
                    patterns_path: str,
                    output_path: str) -> None:
    """
    Preprocessing für das IBM HI-Small Dataset (Kaggle).

    Ziel:
    - Analog zu IBM/Multi-GNN: aus Rohdateien ein ML-fertiges, tabellarisches
      Dataset erzeugen (formatted_transactions.csv).
    - Für diese Bachelorarbeit fokussieren wir auf einen XGBoost-Tabular-Baseline
      und lokale Erklärbarkeit (SHAP), daher:
      * reine ID-Spalten werden aus dem Feature-Space entfernt
      * Zeit- und Betragsinformationen werden in aussagekräftigere Features
        transformiert.

    Parameter
    ----------
    transactions_path : Pfad zu HI-Small_Trans.csv
    patterns_path     : Pfad zu HI-Small_Patterns.txt (derzeit nur für mögliche
                        Erweiterungen geladen, aber nicht aktiv verwendet)
    output_path       : Zielpfad für formatted_transactions.csv
    """

    # 1) Transaktionsdaten laden
    df = pd.read_csv(transactions_path)

    if "Is Laundering" not in df.columns:
        raise ValueError("Spalte 'Is Laundering' nicht im Transaktions-File gefunden.")

    # 2) Timestamp in echtes Datetime konvertieren + einfache Zeitfeatures
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"],
                                         format="%Y/%m/%d %H:%M",
                                         errors="coerce")
        # Basale zeitliche Struktur
        df["Hour"] = df["Timestamp"].dt.hour
        df["Weekday"] = df["Timestamp"].dt.dayofweek
        df["Weekend"] = (df["Weekday"] >= 5).astype(int)
    else:
        # Falls sich das Format ändert, wollen wir explizit scheitern
        raise ValueError("Spalte 'Timestamp' nicht gefunden – bitte Datensatz prüfen.")

    # 3) Einfache relational/strukturelle Flags (Account/Bank/Currency)
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

    # 4) Zielvariable definieren
    #    Kaggle-Spalte "Is Laundering" -> binärer Labelvektor
    df["label"] = df["Is Laundering"].astype(int)

    # 5) Reine ID-Spalten aus dem Feature-Space entfernen
    id_cols = [
        "Account",
        "Account.1",
        "From Bank",
        "To Bank",
    ]

    drop_cols = [c for c in id_cols if c in df.columns]

    # Timestamp als rohe Datetime-Spalte ebenfalls aus dem ML-Feature-Space entfernen,
    # weil wir bereits Hour/Weekday/Weekend extrahiert haben.
    drop_cols.append("Timestamp")

    # Spalten existieren eventuell nicht alle – daher defensive Auswahl:
    drop_cols = [c for c in drop_cols if c in df.columns]

    # Feature-/Target-Split
    target_col = "label"
    feature_df = df.drop(columns=drop_cols + [target_col, "Is Laundering"])

    # 6) One-Hot-Encoding für kategoriale Spalten (Currencies + Payment Format etc.)
    categorical_cols = []
    for col in ["Receiving Currency", "Payment Currency", "Payment Format"]:
        if col in feature_df.columns:
            categorical_cols.append(col)

    feature_df = pd.get_dummies(feature_df, columns=categorical_cols, dummy_na=False)

    # Label wieder anhängen (erste Spalte)
    out_df = pd.concat([df[target_col].reset_index(drop=True), feature_df.reset_index(drop=True)], axis=1)
    out_df.rename(columns={target_col: "label"}, inplace=True)

    # 7) Speichern
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_df.to_csv(output_path, index=False)

    print(f"Formatted dataset saved to: {output_path}")
    print(f"Shape: {out_df.shape}")
    print(f"Positive class share: {out_df['label'].mean():.6f}")

def main():
    parser = argparse.ArgumentParser(description="Format IBM HI-Small Kaggle files into ML-ready CSV.")
    parser.add_argument("--transactions_path", type=str, default="data/raw/HI-Small_Trans.csv")
    parser.add_argument("--patterns_path", type=str, default="data/raw/HI-Small_Patterns.txt")
    parser.add_argument("--output_path", type=str, default="data/processed/formatted_transactions.csv")
    args = parser.parse_args()

    format_hi_small(
        transactions_path=args.transactions_path,
        patterns_path=args.patterns_path,
        output_path=args.output_path,
    )

if __name__ == "__main__":
    main()
