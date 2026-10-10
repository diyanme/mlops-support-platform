from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/customers.csv")

REQUIRED_COLUMNS = {
    "customer_id",
    "tenure_months",
    "monthly_charges",
    "support_tickets",
    "contract_type",
    "churn",
}

ALLOWED_CONTRACT_TYPES = {
    "month-to-month",
    "yearly",
    "two-year",
}


def validate_dataset(data_path: Path) -> bool:
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)

    # 1. Check required columns
    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    # 2. Check that the dataset is not empty
    if df.empty:
        raise ValueError("Dataset is empty")

    # 3. Check missing values
    if df.isnull().any().any():
        missing_values = df.isnull().sum()
        raise ValueError(
            f"Dataset contains missing values:\n{missing_values}"
        )

    # 4. Check target values
    invalid_churn_values = set(df["churn"].unique()) - {0, 1}

    if invalid_churn_values:
        raise ValueError(
            f"Invalid churn values: {sorted(invalid_churn_values)}"
        )

    # 5. Check categorical values
    invalid_contracts = (
        set(df["contract_type"].unique()) - ALLOWED_CONTRACT_TYPES
    )

    if invalid_contracts:
        raise ValueError(
            f"Invalid contract types: {sorted(invalid_contracts)}"
        )

    # 6. Check numeric ranges
    if (df["tenure_months"] < 0).any():
        raise ValueError("tenure_months cannot be negative")

    if (df["monthly_charges"] < 0).any():
        raise ValueError("monthly_charges cannot be negative")

    if (df["support_tickets"] < 0).any():
        raise ValueError("support_tickets cannot be negative")

    print("Dataset validation passed.")
    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")
    print(f"Churn distribution:\n{df['churn'].value_counts()}")

    return True


if __name__ == "__main__":
    validate_dataset(DATA_PATH)