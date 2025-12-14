# src/train.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
import joblib
import mlflow

# -----------------------------
# Load processed data
# -----------------------------
data_file = "data/processed/processed_data_with_risk.csv"
df = pd.read_csv(data_file)

# -----------------------------
# Select numeric features
# -----------------------------
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
numeric_cols.remove("is_high_risk")  # Remove target from features

X = df[numeric_cols]
y = df["is_high_risk"]

# -----------------------------
# Impute missing values
# -----------------------------
imputer = SimpleImputer(strategy="mean")
X_imputed = imputer.fit_transform(X)

# -----------------------------
# Train/test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# MLflow setup
# -----------------------------
mlflow.set_experiment("credit-risk-model")

with mlflow.start_run():
    # -----------------------------
    # Train model
    # -----------------------------
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)

    # -----------------------------
    # Evaluate
    # -----------------------------
    y_pred = lr.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc}")
    print(classification_report(y_test, y_pred))

    # -----------------------------
    # Save model
    # -----------------------------
    model_path = "models/logistic_regression_model.pkl"
    joblib.dump(lr, model_path)
    print(f"Model saved to {model_path}")

    # -----------------------------
    # Log with MLflow
    # -----------------------------
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(lr, "model")
