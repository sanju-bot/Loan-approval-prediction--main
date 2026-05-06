#!/usr/bin/env python
"""
Simple Model Training - Generates fresh model files compatible with current environment
Run this ONCE to create best_model.pkl and scaler.pkl
"""
import os
import sys
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

print("\n" + "=" * 70)
print("🏦 LOAN APPROVAL PREDICTION - SIMPLE MODEL TRAINING")
print("=" * 70)

try:
    # Step 1: Load data
    print("\n📥 Step 1: Loading data...")
    df = pd.read_csv('Loan_Data.csv')
    print(f"✓ Loaded {df.shape[0]} records with {df.shape[1]} columns")
    
    # Clean column names
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    print(f"✓ Columns: {df.columns.tolist()}")
    
    # Step 2: Prepare data
    print("\n📊 Step 2: Preparing data...")
    target_col = 'loan_status'
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found!")
    
    X = df.drop([target_col, 'loan_id'], axis=1, errors='ignore')
    y = df[target_col]
    
    print(f"✓ Features: {X.shape[1]}")
    print(f"✓ Target distribution:\n{y.value_counts()}")
    
    # Step 3: Encode categorical variables BEFORE splitting
    print("\n🔄 Step 3: Encoding categorical variables...")
    label_encoders = {}
    
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    print(f"  Categorical columns: {categorical_cols}")
    
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str).str.strip())
        label_encoders[col] = le
        print(f"  ✓ Encoded {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
    # Encode target - CRITICAL: Approved=1, Rejected=0
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y.astype(str).str.strip())
    print(f"  ✓ Target encoding: {dict(zip(le_target.classes_, le_target.transform(le_target.classes_)))}")
    
    # Step 4: Split data
    print("\n📋 Step 4: Splitting data (80-20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )
    print(f"✓ Training: {X_train.shape[0]} records")
    print(f"✓ Testing: {X_test.shape[0]} records")
    
    # Step 5: Scale features
    print("\n🔧 Step 5: Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("✓ Features scaled")
    
    # Step 6: Train Random Forest (more stable than Decision Tree)
    print("\n🤖 Step 6: Training Random Forest...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✓ Accuracy: {accuracy:.4f}")
    
    # Step 7: Detailed metrics
    print("\n📊 Detailed Metrics:")
    print(f"  Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"  Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"  F1-Score:  {f1_score(y_test, y_pred):.4f}")
    
    # Step 8: Save model and scaler
    print("\n💾 Step 8: Saving model and scaler...")
    os.makedirs('models', exist_ok=True)
    
    joblib.dump(model, 'best_model.pkl')
    joblib.dump(model, 'models/best_model.pkl')
    print("✓ Model saved to best_model.pkl and models/best_model.pkl")
    
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    print("✓ Scaler saved to scaler.pkl and models/scaler.pkl")
    
    # Save encoders for reference
    joblib.dump(label_encoders, 'label_encoders.pkl')
    print("✓ Label encoders saved to label_encoders.pkl")
    
    print("\n" + "=" * 70)
    print("✅ MODEL TRAINING COMPLETE!")
    print("=" * 70)
    print("\nNow you can run: python app.py")
    print("=" * 70 + "\n")
    
except Exception as e:
    print(f"\n❌ ERROR: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
