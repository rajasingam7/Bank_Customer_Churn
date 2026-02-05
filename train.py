import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from xgboost import XGBClassifier

from src.feature_engineering import create_features


# --------------------------------------------------
# Paths
# --------------------------------------------------
DATA_PATH = "data/churn.csv"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)


# --------------------------------------------------
# Load Dataset
# --------------------------------------------------
df = pd.read_csv(DATA_PATH)

print("Columns in dataset:", df.columns.tolist())


# --------------------------------------------------
# Rename Columns (Match ML Pipeline)
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
# Remove Non Predictive Columns
# --------------------------------------------------
if "customer_id" in df.columns:
    df.drop("customer_id", axis=1, inplace=True)


# --------------------------------------------------
# Target Column
# --------------------------------------------------
target_col = "churn"


# --------------------------------------------------
# Encoding
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


# --------------------------------------------------
# Scaling
# --------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# --------------------------------------------------
# Train Test Split
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)


# --------------------------------------------------
# ML Models
# --------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(probability=True),
    "XGBoost": XGBClassifier(eval_metric="logloss")
}


results = []


# --------------------------------------------------
# Train Models
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

    # Save each trained model
    joblib.dump(model, f"{MODEL_DIR}/{name}.pkl")


# --------------------------------------------------
# Save Evaluation Results
# --------------------------------------------------
results_df = pd.DataFrame(results)
results_df.to_csv(f"{MODEL_DIR}/model_results.csv", index=False)

joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
joblib.dump(encoders, f"{MODEL_DIR}/encoders.pkl")


print("\n✅ Training Completed Successfully")
print(results_df)
