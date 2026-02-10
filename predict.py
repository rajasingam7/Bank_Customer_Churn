import pandas as pd
import joblib
import os

from src.feature_engineering import create_features


# --------------------------------------------------
# Base directory (CRITICAL for AWS / Linux)
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "models")

SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "encoders.pkl")
FEATURE_COLS_PATH = os.path.join(MODEL_DIR, "feature_columns.pkl")


def predict_churn(payload, model_name):
    """
    Predict churn probability for a single customer
    """

    # --------------------------------------------------
    # Load artifacts
    # --------------------------------------------------
    model_path = os.path.join(MODEL_DIR, f"{model_name}.pkl")

    model = joblib.load(model_path)
    scaler = joblib.load(SCALER_PATH)
    encoders = joblib.load(ENCODER_PATH)
    feature_columns = joblib.load(FEATURE_COLS_PATH)

    # --------------------------------------------------
    # Create input DataFrame
    # --------------------------------------------------
    df = pd.DataFrame([payload])

    # Feature engineering
    df = create_features(df)

    # --------------------------------------------------
    # Encoding categorical columns
    # --------------------------------------------------
    for col in ["Geography", "Gender"]:
        df[col] = encoders[col].transform(df[col])

    # --------------------------------------------------
    # Align columns EXACTLY as training
    # --------------------------------------------------
    df = df[feature_columns]

    # --------------------------------------------------
    # Scaling
    # --------------------------------------------------
    df_scaled = scaler.transform(df)

    # --------------------------------------------------
    # Prediction (probability of churn)
    # --------------------------------------------------
    prob = model.predict_proba(df_scaled)[0][1]

    return prob
