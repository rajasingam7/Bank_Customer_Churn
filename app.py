import streamlit as st
import pandas as pd

from predict import predict_churn
from src.feature_engineering import create_features


# --------------------------------------------------
# Page Settings
# --------------------------------------------------
st.set_page_config(page_title="Churn Predictor", layout="wide")

st.title("🏦 Bank Customer Churn Prediction Dashboard")


# --------------------------------------------------
# Load Dataset
# --------------------------------------------------
DATA_PATH = "data/churn.csv"


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


df = load_data()

# Feature engineered version
df_fe = create_features(df.copy())


# --------------------------------------------------
# Sidebar Navigation
# --------------------------------------------------
menu = st.sidebar.radio(
    "Select Option",
    [
        "Prediction",
        "View Raw Data",
        "View Feature Engineered Data",
        "Model Comparison"
    ]
)


# --------------------------------------------------
# RAW DATA VIEW
# --------------------------------------------------
if menu == "View Raw Data":

    st.subheader("📊 Raw Dataset")

    st.write("Dataset Shape:", df.shape)
    st.dataframe(df)

    st.subheader("Column Names")
    st.write(df.columns.tolist())


# --------------------------------------------------
# FEATURE ENGINEERING VIEW
# --------------------------------------------------
elif menu == "View Feature Engineered Data":

    st.subheader("⚙️ Feature Engineered Dataset")

    st.write("Dataset Shape:", df_fe.shape)
    st.dataframe(df_fe)

    new_cols = list(set(df_fe.columns) - set(df.columns))

    st.subheader("🆕 New Columns Added")
    st.write(new_cols)

    st.write("Original Column Count:", len(df.columns))
    st.write("After Feature Engineering:", len(df_fe.columns))


# --------------------------------------------------
# MODEL COMPARISON DASHBOARD
# --------------------------------------------------
elif menu == "Model Comparison":

    st.subheader("📊 ML Model Performance Comparison")

    try:
        results = pd.read_csv("models/model_results.csv")

        st.dataframe(results)

        st.subheader("📈 Model Metrics Chart")
        st.bar_chart(results.set_index("Model"))

    except Exception:
        st.warning("⚠️ model_results.csv not found. Run train.py first.")


# --------------------------------------------------
# PREDICTION UI
# --------------------------------------------------
elif menu == "Prediction":

    st.subheader("Enter Customer Details")

    model_choice = st.selectbox(
        "Select ML Model",
        ["Logistic Regression", "Random Forest", "SVM", "XGBoost"]
    )

    col1, col2 = st.columns(2)

    with col1:

        geography = st.selectbox("Geography", ["France", "Germany", "Spain"])

        gender = st.selectbox("Gender", ["Male", "Female"])

        credit_score = st.number_input("Credit Score", 300, 900, 650)

        age = st.number_input("Age", 18, 100, 35)

        tenure = st.number_input("Tenure", 0, 10, 3)

    with col2:

        balance = st.number_input("Balance", 0.0, 250000.0, 50000.0)

        salary = st.number_input("Estimated Salary", 0.0, 200000.0, 60000.0)

        products = st.selectbox("Number of Products", [1, 2, 3, 4])

        has_card = st.selectbox("Has Credit Card", [0, 1])

        is_active = st.selectbox("Is Active Member", [0, 1])

    if st.button("Predict"):

        payload = {
            "Geography": geography,
            "Gender": gender,
            "CreditScore": credit_score,
            "Age": age,
            "Tenure": tenure,
            "Balance": balance,
            "NumOfProducts": products,
            "HasCrCard": has_card,
            "IsActiveMember": is_active,
            "EstimatedSalary": salary
        }

        try:
            prob = predict_churn(payload, model_choice)

            if prob >= 0.5:
                st.error("⚠️ Customer is likely to LEAVE")
            else:
                st.success("✅ Customer is likely to CONTINUE")

            st.write(f"Churn Probability: {prob:.2%}")
            st.progress(int(prob * 100))

            metrics_df = pd.read_csv("models/model_results.csv")
            selected_metrics = metrics_df[metrics_df["Model"] == model_choice]

            st.subheader("Model Performance Metrics")
            st.dataframe(selected_metrics)

        except Exception as e:
            st.error(str(e))
