import os
import sys
import pandas as pd
import joblib
from data_preprocessing import preprocess_data
from eda import perform_eda, print_statistics
from model_training import (prepare_features_target, split_data,
                            scale_features, train_models,
                            tune_decision_tree, tune_random_forest,
                            perform_cross_validation)
from model_evaluation import evaluate_models, get_best_model, plot_confusion_matrix
from prediction import save_model, load_model, test_prediction, predict_custom_applicant

# Create directories if they don't exist
os.makedirs('output/plots', exist_ok=True)
os.makedirs('output/reports', exist_ok=True)  # ✅ ADD THIS
os.makedirs('models', exist_ok=True)


def main():
    try:
        print("\n" + "=" * 70)
        print("🏦 LOAN APPROVAL PREDICTION SYSTEM")
        print("=" * 70)
        sys.stdout.flush()

        # Step 1: Preprocess Data
        print("\n📥 Step 1: Data Preprocessing")
        sys.stdout.flush()
        df = preprocess_data('Loan_Data.csv')

        # ✅ DEBUG: Show columns
        print("\n🔍 DEBUG: Columns in DataFrame:")
        print(df.columns.tolist())
        print(f"\nDataFrame Shape: {df.shape}")
        print(f"\nFirst 5 rows:\n{df.head()}")
        sys.stdout.flush()

        print_statistics(df)

        # Step 2: EDA
        print("\n📈 Step 2: Exploratory Data Analysis")
        sys.stdout.flush()
        perform_eda(df)

        # Step 3: Prepare Features
        print("\n🎯 Step 3: Preparing Features")
        sys.stdout.flush()
        X, y = prepare_features_target(df)
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        sys.stdout.flush()

        # Step 4: Split Data
        print("\n📊 Step 4: Splitting Data (80-20)")
        sys.stdout.flush()
        X_train, X_test, y_train, y_test = split_data(X, y)
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        sys.stdout.flush()

        X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

        # ✅ Save scaler
        joblib.dump(scaler, 'models/scaler.pkl')
        print("✓ Scaler saved to models/scaler.pkl")

        # Step 5: Hyperparameter Tuning
        print("\n🔧 Step 5: Hyperparameter Tuning")
        sys.stdout.flush()
        print("⚠️  This may take a few minutes...")

        best_dt = tune_decision_tree(X_train_scaled, y_train)
        best_rf = tune_random_forest(X_train_scaled, y_train)

        # Cross-validation
        dt_cv_scores = perform_cross_validation(best_dt, X_train_scaled, y_train)
        rf_cv_scores = perform_cross_validation(best_rf, X_train_scaled, y_train)

        # Use tuned models
        trained_models = {
            'Tuned Decision Tree': best_dt,
            'Tuned Random Forest': best_rf
        }
        # Step 6: Evaluate Models
        print("\n📊 Step 6: Evaluating Models")
        sys.stdout.flush()
        results = evaluate_models(trained_models, X_test_scaled, y_test)

        # Step 7: Select Best Model
        best_model_name = get_best_model(results)
        best_model = trained_models[best_model_name]
        print(f"\n🏆 Best Model: {best_model_name}")
        sys.stdout.flush()

        # Step 8: Plot Confusion Matrix
        print("\n📊 Step 8: Plotting Confusion Matrix")
        sys.stdout.flush()
        plot_confusion_matrix(best_model, X_test_scaled, y_test, best_model_name)

        # Step 9: Save Model
        print("\n💾 Step 9: Saving Model")
        sys.stdout.flush()
        save_model(best_model)

        # Step 10: Test Prediction
        print("\n✅ Step 10: Testing Prediction on New Data")
        sys.stdout.flush()
        test_prediction(best_model, scaler, X.columns)

        # Step 11: Test Multiple Scenarios
        print("\n🧪 Step 11: Testing Multiple Loan Scenarios:")
        print("=" * 70)

        test_cases = [
            {
                'name': 'High Income, High Credit Score',
                'income': 8000000,
                'loan_amount': 300000,
                'cibil_score': 800,
                'no_of_dependents': 2
            },
            {
                'name': 'Low Income, Low Credit Score',
                'income': 2000000,
                'loan_amount': 500000,
                'cibil_score': 550,
                'no_of_dependents': 4
            },
            {
                'name': 'Medium Income, Good Credit',
                'income': 5000000,
                'loan_amount': 200000,
                'cibil_score': 700,
                'no_of_dependents': 1
            }
        ]

        for case in test_cases:
            print(f"\n📝 {case['name']}:")
            result = predict_custom_applicant(
                best_model, scaler,
                income=case['income'],
                loan_amount=case['loan_amount'],
                cibil_score=case['cibil_score'],
                no_of_dependents=case['no_of_dependents']
            )
            print(f"   Result: {result['result']}")
            print(f"   Confidence: {result['confidence']}")

        print("\n" + "=" * 70)
        print("✅ PIPELINE COMPLETE!")
        print("=" * 70 + "\n")
        sys.stdout.flush()

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()


if __name__ == "__main__":
    main()
    input("\n👉 Press ENTER to close the program...")