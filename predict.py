import pandas as pd
import joblib
from src.feature_engineering import create_features


MODEL_DIR = "models"


def predict_churn(payload, model_name):

    model = joblib.load(f"{MODEL_DIR}/{model_name}.pkl")
    scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")
    encoders = joblib.load(f"{MODEL_DIR}/encoders.pkl")
    feature_columns = joblib.load(f"{MODEL_DIR}/feature_columns.pkl")

    df = pd.DataFrame([payload])

    # Feature engineering
    df = create_features(df)

    # Encoding
    for col in ["Geography", "Gender"]:
        df[col] = encoders[col].transform(df[col])

    # ⭐ Align feature order EXACTLY like training
    df = df[feature_columns]

    # Scaling
    df_scaled = scaler.transform(df)

    prob = model.predict_proba(df_scaled)[0][1]

    return prob
