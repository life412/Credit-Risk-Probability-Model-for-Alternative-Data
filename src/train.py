import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load processed data with risk label
data_file = "data/processed/processed_data_with_risk.csv"
df = pd.read_csv(data_file)
df = df.dropna()

# Select numeric columns only for training
numeric_cols = [
    'Amount','Value','total_transaction_amount','avg_transaction_amount',
    'transaction_count','std_transaction_amount','transaction_hour',
    'transaction_day','transaction_month','transaction_year',
    'PricingStrategy','FraudResult'
]

X = df[numeric_cols]
y = df['is_high_risk']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train logistic regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

# Make predictions
y_pred = lr.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save trained model
model_file = "models/logistic_regression_model.pkl"
joblib.dump(lr, model_file)
print(f"Model saved to {model_file}")
