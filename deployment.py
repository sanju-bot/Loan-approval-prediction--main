import joblib
import pandas as pd

# Load model and scaler
model = joblib.load('models/best_model.pkl')
scaler = joblib.load('models/scaler.pkl')


def predict_loan(income, loan_amount, cibil_score, dependents=2):
    """
    Predict loan approval

    Example:
    >>> predict_loan(5000000, 200000, 750, 2)
    {'result': '✅ APPROVED', 'confidence': '99.95%'}
    """
    from src.prediction import predict_custom_applicant
    return predict_custom_applicant(model, scaler, income, loan_amount, cibil_score, dependents)


# Test
if __name__ == '__main__':
    result = predict_loan(6000000, 250000, 800, 2)
    print(result)