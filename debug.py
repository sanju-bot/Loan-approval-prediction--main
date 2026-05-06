from src.data_preprocessing import preprocess_data

# Load and preprocess
df = preprocess_data('data/Loan_Data.csv')

# Show everything
print("\n" + "="*60)
print("✅ FINAL DATAFRAME CHECK")
print("="*60)
print(f"Shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nData Types:\n{df.dtypes}")
print(f"\nFirst 5 rows:\n{df.head()}")
print(f"\nLast 5 rows:\n{df.tail()}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print("="*60)

# Try to prepare features
try:
    from src.model_training import prepare_features_target
    X, y = prepare_features_target(df)
    print("\n✅ Features & Target prepared successfully!")
except Exception as e:
    print(f"\n❌ ERROR: {e}")