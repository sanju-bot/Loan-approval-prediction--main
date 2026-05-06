#!/usr/bin/env python
"""
Quick test script to validate project setup
"""
import os
import sys

# Change to script directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

print("=" * 70)
print("🏦 LOAN APPROVAL PREDICTION - PROJECT VALIDATION")
print("=" * 70)

# Test 1: Check if data file exists
print("\n✓ Test 1: Checking data file...")
if os.path.exists('Loan_Data.csv'):
    print("  ✅ Loan_Data.csv found!")
else:
    print("  ❌ Loan_Data.csv NOT found!")
    sys.exit(1)

# Test 2: Check imports
print("\n✓ Test 2: Checking imports...")
try:
    import pandas as pd
    print("  ✅ pandas imported")
    import numpy as np
    print("  ✅ numpy imported")
    from sklearn.preprocessing import StandardScaler
    print("  ✅ sklearn imported")
    import matplotlib.pyplot as plt
    print("  ✅ matplotlib imported")
    import seaborn as sns
    print("  ✅ seaborn imported")
    import joblib
    print("  ✅ joblib imported")
except ImportError as e:
    print(f"  ❌ Import error: {e}")
    sys.exit(1)

# Test 3: Check module imports
print("\n✓ Test 3: Checking module imports...")
try:
    from data_preprocessing import preprocess_data
    print("  ✅ data_preprocessing imported")
    from eda import perform_eda, print_statistics
    print("  ✅ eda imported")
    from model_training import (prepare_features_target, split_data,
                                scale_features, train_models,
                                tune_decision_tree, tune_random_forest,
                                perform_cross_validation)
    print("  ✅ model_training imported")
    from model_evaluation import evaluate_models, get_best_model, plot_confusion_matrix
    print("  ✅ model_evaluation imported")
    from prediction import save_model, load_model, test_prediction, predict_custom_applicant
    print("  ✅ prediction imported")
except ImportError as e:
    print(f"  ❌ Module import error: {e}")
    sys.exit(1)

# Test 4: Load data
print("\n✓ Test 4: Loading data...")
try:
    df = preprocess_data('Loan_Data.csv')
    print(f"  ✅ Data loaded successfully! Shape: {df.shape}")
    print(f"  Columns: {df.columns.tolist()}")
except Exception as e:
    print(f"  ❌ Error loading data: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("✅ ALL VALIDATION TESTS PASSED!")
print("=" * 70)
print("\nYou can now run: python main.py")
