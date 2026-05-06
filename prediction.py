import pandas as pd
import joblib


# ❌ DELETE THIS LINE:
# from src.prediction import load_model, predict_custom_applicant

def predict_loan_approval(model, new_applicant_data):
    """Predict loan approval for new applicant"""
    prediction = model.predict(new_applicant_data)
    probability = model.predict_proba(new_applicant_data)

    result = "✅ APPROVED" if prediction[0] == 1 else "❌ REJECTED"
    confidence = max(probability[0]) * 100

    return result, confidence


def save_model(model, filepath='models/best_model.pkl'):
    """Save trained model"""
    joblib.dump(model, filepath)
    print(f"✓ Model saved to {filepath}")


def load_model(filepath='models/best_model.pkl'):
    """Load trained model"""
    return joblib.load(filepath)


def test_prediction(model, scaler, feature_names):
    """Test prediction with sample data"""
    print("\n🧪 Testing Prediction with Sample Applicant Data:")
    print("=" * 60)

    # ✅ Use ACTUAL column names from your dataset (WITHOUT loan_id)
    sample_applicant = pd.DataFrame({
        'no_of_dependents': [2],
        'education': [0],  # 0 = Graduate, 1 = Not Graduate
        'self_employed': [0],  # 0 = No, 1 = Yes
        'income_annum': [5000000],  # 50 lakh
        'loan_amount': [200000],  # 2 lakh
        'loan_term': [360],  # 30 years
        'cibil_score': [750],  # Good credit score
        'residential_assets_value': [3000000],
        'commercial_assets_value': [1000000],
        'luxury_assets_value': [500000],
        'bank_asset_value': [5000000]
    })

    print("\n📋 Sample Applicant Data:")
    print(sample_applicant)

    # ✅ Scale the data (no need to drop loan_id, it's not here)
    sample_scaled = scaler.transform(sample_applicant)

    # ✅ Predict
    result, confidence = predict_loan_approval(model, sample_scaled)

    print(f"\n🎯 Prediction Result: {result}")
    print(f"📊 Confidence: {confidence:.2f}%")
    print("=" * 60)


# ✅ BONUS: Function to predict with custom data
def predict_custom_applicant(model, scaler, income, loan_amount, cibil_score,
                             no_of_dependents=2, education=0, self_employed=0, loan_term=360,
                             residential_assets=3000000, commercial_assets=1000000,
                             luxury_assets=500000, bank_assets=5000000):
    """
    Predict loan approval for custom applicant

    Parameters:
    - income: Annual income (e.g., 5000000)
    - loan_amount: Loan amount (e.g., 200000)
    - cibil_score: Credit score (e.g., 750)
    - no_of_dependents: Number of dependents (0-5)
    - education: 0 = Graduate, 1 = Not Graduate
    - self_employed: 0 = No, 1 = Yes
    - loan_term: Loan term in months
    - residential_assets: Residential assets value
    - commercial_assets: Commercial assets value
    - luxury_assets: Luxury assets value
    - bank_assets: Bank assets value
    """

    applicant = pd.DataFrame({
        'no_of_dependents': [no_of_dependents],
        'education': [education],
        'self_employed': [self_employed],
        'income_annum': [income],
        'loan_amount': [loan_amount],
        'loan_term': [loan_term],
        'cibil_score': [cibil_score],
        'residential_assets_value': [residential_assets],
        'commercial_assets_value': [commercial_assets],
        'luxury_assets_value': [luxury_assets],
        'bank_asset_value': [bank_assets]
    })

    # Scale
    applicant_scaled = scaler.transform(applicant)

    # Predict
    result, confidence = predict_loan_approval(model, applicant_scaled)

    return {
        'result': result,
        'confidence': f"{confidence:.2f}%",
        'prediction_code': 1 if '✅' in result else 0
    }
