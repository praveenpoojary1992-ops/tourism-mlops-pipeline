"""Data preparation: load from the repo, clean, split and save train/test files locally.

The four CSV files are written to the current working directory; the GitHub Actions
workflow uploads them as an artifact for the model-training job.
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "tourism.csv"
TARGET = "ProdTaken"


def main():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {DATA_PATH} -> shape {df.shape}")

    # 1. Remove unnecessary columns (row index and unique customer identifier)
    drop_cols = [c for c in ["Unnamed: 0", "CustomerID"] if c in df.columns]
    df = df.drop(columns=drop_cols)
    print(f"Dropped columns: {drop_cols}")

    # 2. Clean inconsistent category labels
    text_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in text_cols:
        df[col] = df[col].str.strip()
    df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})
    df["MaritalStatus"] = df["MaritalStatus"].replace({"Unmarried": "Single"})

    # 3. Remove duplicates (missing values are imputed inside the model pipeline)
    before = len(df)
    df = df.drop_duplicates()
    print(f"Removed {before - len(df)} duplicate rows")
    print(f"Missing values remaining: {int(df.isna().sum().sum())}")

    # 4. Split into train and test sets (stratified on the imbalanced target)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5. Save locally
    Xtrain.to_csv("Xtrain.csv", index=False)
    Xtest.to_csv("Xtest.csv", index=False)
    ytrain.to_csv("ytrain.csv", index=False)
    ytest.to_csv("ytest.csv", index=False)

    print(f"\nXtrain: {Xtrain.shape} | Xtest: {Xtest.shape}")
    print(f"Buyers in train: {ytrain.mean():.3f} | Buyers in test: {ytest.mean():.3f}")
    print("Saved: Xtrain.csv, Xtest.csv, ytrain.csv, ytest.csv")


if __name__ == "__main__":
    main()
