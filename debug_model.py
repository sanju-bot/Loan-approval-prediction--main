#!/usr/bin/env python
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

print("\n=== MODEL DEBUG INFO ===\n")

# Load data
df = pd.read_csv('Loan_Data.csv')
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

# Check encoding
le = LabelEncoder()
y_encoded = le.fit_transform(df['loan_status'].astype(str).str.strip())

print("ENCODING:")
print(f"  Classes: {le.classes_}")
print(f"  'Approved' encoded as: {le.transform(['Approved'])[0]}")
print(f"  'Rejected' encoded as: {le.transform(['Rejected'])[0]}")

# Load model
model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')

# Test the input
test_data = pd.DataFrame({
    'no_of_dependents': [2],
    'education': [1],
    'self_employed': [0],
    'income_annum': [7500000],
    'loan_amount': [250000],
    'loan_term': [360],
    'cibil_score': [780],
    'residential_assets_value': [4000000],
    'commercial_assets_value': [1500000],
    'luxury_assets_value': [600000],
    'bank_asset_value': [800000]
})

scaled = scaler.transform(test_data)
pred = model.predict(scaled)[0]
proba = model.predict_proba(scaled)[0]

print("\nPREDICTION:")
print(f"  Raw prediction value: {pred}")
print(f"  Probability for class 0 (Rejected): {proba[0]:.4f}")
print(f"  Probability for class 1 (Approved): {proba[1]:.4f}")
print(f"  Predicted class: {pred}")

if pred == 1:
    print("\n  ✅ Should be APPROVED")
elif pred == 0:
    print("\n  ❌ Should be REJECTED")

print("\n=== END DEBUG ===\n")
