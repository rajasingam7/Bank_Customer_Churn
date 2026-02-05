import pandas as pd

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Safe guards if columns are missing
    if "Balance" in df and "EstimatedSalary" in df:
        df["BalanceSalaryRatio"] = df["Balance"] / (df["EstimatedSalary"] + 1)

    if "Age" in df:
        df["AgeGroup"] = (df["Age"] // 10).astype(int)

    if "NumOfProducts" in df and "Tenure" in df:
        df["ProductsPerTenure"] = df["NumOfProducts"] / (df["Tenure"] + 1)

    return df
