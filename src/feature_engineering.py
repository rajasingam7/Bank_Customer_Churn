import pandas as pd


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create engineered features for churn prediction.
    Safe for both training and inference.
    """

    df = df.copy()

    # --------------------------------------------------
    # Balance / Salary ratio
    # --------------------------------------------------
    if {"Balance", "EstimatedSalary"}.issubset(df.columns):
        df["BalanceSalaryRatio"] = df["Balance"] / (df["EstimatedSalary"] + 1)
    else:
        df["BalanceSalaryRatio"] = 0

    # --------------------------------------------------
    # Age group (decades)
    # --------------------------------------------------
    if "Age" in df.columns:
        df["AgeGroup"] = (df["Age"] // 10).astype(int)
    else:
        df["AgeGroup"] = 0

    # --------------------------------------------------
    # Products per tenure
    # --------------------------------------------------
    if {"NumOfProducts", "Tenure"}.issubset(df.columns):
        df["ProductsPerTenure"] = df["NumOfProducts"] / (df["Tenure"] + 1)
    else:
        df["ProductsPerTenure"] = 0

    return df
