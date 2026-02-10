import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.feature_engineering import create_features


# --------------------------------------------------
# Base Directory (IMPORTANT)
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "data", "churn.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)


# --------------------------------------------------
# Load Dataset
# --------------------------------------------------
df = pd.read_csv(DATA_PATH)
print("Columns in dataset:", df.columns.tolist())


# --------------------------------------------------
# Rename Columns (align with inference payload)
# --------------------------------------------------
df.rename(columns={
    "country": "Geography",
    "products_number": "NumOfProducts",
    "credit_card": "HasCrCard",
    "active_member": "IsActiveMember",
    "estimated_salary": "EstimatedSalary",
    "credit_score": "CreditScore",
    "age": "Age",
    "tenure": "Tenure",
    "balance": "Balance",
    "gender": "Gender"
}, inplace=True)


# --------------------------------------------------
# Feature Engineering
# --------------------------------------------------
df = create_features(df)


# --------------------------------------------------
# Remove Non-Predictive Columns
# --------------------------------------------------
if "customer_id" in df.columns:
    df.drop("customer_id", axis=1, inplace=True)


# --------------------------------------------------
# Target Column
# --------------------------------------------------
target_col = "churn"


# --------------------------------------------------
# Encoding (categorical)
# --------------------------------------------------
encoders = {}

for col in ["Geography", "Gender"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le


# --------------------------------------------------
# Split Features / Target
# --------------------------------------------------
X = df.drop(target_col, axis=1)
y = df[target_col]

# ⭐ Save feature order (CRITICAL for inference)
feature_columns = X.columns.tolist()
joblib.dump(feature_columns, os.path.join(MODEL_DIR, "feature_columns.pkl"))


# --------------------------------------------------
# Scaling
# --------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# --------------------------------------------------
# Train-Test Split
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# Models
# --------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "XGBoost": XGBClassifier(
        eval_metric="logloss",
        random_state=42,
        use_label_encoder=False
    )
}


results = []


# --------------------------------------------------
# Train & Evaluate
# --------------------------------------------------
for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, preds),
        "Precision": precision_score(y_test, preds),
        "Recall": recall_score(y_test, preds),
        "F1 Score": f1_score(y_test, preds),
        "ROC AUC": roc_auc_score(y_test, probs)
    })

    # Save trained model
    joblib.dump(model, os.path.join(MODEL_DIR, f"{name}.pkl"))


# --------------------------------------------------
# Save Artifacts
# --------------------------------------------------
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(MODEL_DIR, "model_results.csv"), index=False)

joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
joblib.dump(encoders, os.path.join(MODEL_DIR, "encoders.pkl"))


print("\n✅ Training Completed Successfully")
print(results_df)
