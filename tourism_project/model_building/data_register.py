"""Data registration: validate the tourism dataset stored in the repo and print a summary."""
import sys
from pathlib import Path

import pandas as pd

# Resolve the path from this file so the script works from any working directory
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "tourism.csv"

EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier", "DurationOfPitch",
    "Occupation", "Gender", "NumberOfPersonVisiting", "NumberOfFollowups",
    "ProductPitched", "PreferredPropertyStar", "MaritalStatus", "NumberOfTrips",
    "Passport", "PitchSatisfactionScore", "OwnCar", "NumberOfChildrenVisiting",
    "Designation", "MonthlyIncome",
]


def main():
    if not DATA_PATH.exists():
        print(f"Validation FAILED - dataset not found at {DATA_PATH}")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    if missing:
        print(f"Validation FAILED - missing expected columns: {missing}")
        sys.exit(1)

    print("Validation PASSED - all expected columns are present.")
    if extra:
        print(f"Extra columns found (will be dropped in data preparation): {extra}")

    print(f"\nDataset registered from: {DATA_PATH}")
    print(f"Rows: {df.shape[0]}  |  Columns: {df.shape[1]}")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    print("\nMissing values per column (only columns with missing values):")
    na = df.isna().sum()
    print(na[na > 0] if (na > 0).any() else "None")
    print("\nTarget distribution (ProdTaken):")
    print(df["ProdTaken"].value_counts(normalize=True).round(3).to_string())
    print("\nData types:")
    print(df.dtypes.to_string())


if __name__ == "__main__":
    main()
