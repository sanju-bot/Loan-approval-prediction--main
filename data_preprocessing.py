import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_data(filepath):
    """Load dataset from CSV"""
    df = pd.read_csv(filepath)
    print(f"✓ Data loaded. Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    return df

def find_target_column(df):
        """Automatically find the target column"""
        possible_names = ['loan_status', 'Loan_Status', 'Status', 'status',
                          'Approval', 'approval', 'Approved', 'approved',
                          'target', 'class', 'label']

        for col in possible_names:
            if col in df.columns:
                print(f"✓ Found target column: '{col}'")
                return col

        raise ValueError(f"❌ Target column not found! Available columns: {df.columns.tolist()}")
def handle_missing_values(df):
    """Fill missing values with mode for categorical, median for numerical"""
    print("Handling missing values...")

    # Show missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"Missing values found:\n{missing[missing > 0]}")

    # Categorical columns - fill with mode
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols:
        if df[col].isnull().sum() > 0:
            mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown'
            df[col].fillna(mode_val, inplace=True)
            print(f"  ✓ Filled {col} with mode: {mode_val}")

    # Numerical columns - fill with median
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    for col in numerical_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"  ✓ Filled {col} with median: {median_val}")

    return df


def encode_categorical(df, target_column):
    """Encode categorical variables to numerical"""
    print("Encoding categorical variables...")

    # Get all object (string) columns except target
    object_cols = df.select_dtypes(include=['object']).columns.tolist()

    for col in object_cols:
        if col == target_column:
            continue  # Don't encode target yet

        # Get unique values
        unique_vals = df[col].unique()
        print(f"  • {col}: {list(unique_vals)}")

        # Create mapping based on unique values
        if set(unique_vals).issubset({'Yes', 'No', 'yes', 'no'}):
            mapping = {'Yes': 1, 'No': 0, 'yes': 1, 'no': 0}
        elif set(unique_vals).issubset({'Male', 'Female', 'male', 'female'}):
            mapping = {'Male': 1, 'Female': 0, 'male': 1, 'female': 0}
        elif set(unique_vals).issubset({'Urban', 'Rural', 'Semi-Urban', 'urban', 'rural', 'semi-urban', 'Semi-urban'}):
            mapping = {'Urban': 2, 'Semi-Urban': 1, 'Semi-urban': 1, 'Rural': 0, 'urban': 2, 'rural': 0,
                       'semi-urban': 1}
        elif set(unique_vals).issubset({'Graduate', 'Not Graduate', 'graduate', 'not graduate'}):
            mapping = {'Graduate': 1, 'Not Graduate': 0, 'graduate': 1, 'not graduate': 0}
        else:
            # For other columns, use label encoding
            mapping = {val: idx for idx, val in enumerate(sorted(unique_vals))}

        df[col] = df[col].map(mapping)
        print(f"    ✓ Encoded with mapping: {mapping}")

    # Encode target column
    print(f"\nEncoding target column: {target_column}")
    target_unique = df[target_column].unique()
    print(f"  Unique values in target: {list(target_unique)}")

    # Auto-detect target encoding
    if set(target_unique).issubset({'Y', 'N', 'y', 'n'}):
        target_mapping = {'Y': 1, 'N': 0, 'y': 1, 'n': 0}
    elif set(target_unique).issubset({0, 1, '0', '1'}):
        target_mapping = {0: 0, 1: 1, '0': 0, '1': 1}
    elif set(target_unique).issubset({'Yes', 'No', 'yes', 'no'}):
        target_mapping = {'Yes': 1, 'No': 0, 'yes': 1, 'no': 0}
    elif set(target_unique).issubset({'Approved', 'Rejected', 'approved', 'rejected'}):
        target_mapping = {'Approved': 1, 'Rejected': 0, 'approved': 1, 'rejected': 0}
    else:
        target_mapping = {val: (1 if idx > 0 else 0) for idx, val in enumerate(sorted(target_unique))}

    df[target_column] = df[target_column].map(target_mapping)
    print(f"  ✓ Target encoded: {target_mapping}")

    return df


def drop_unnecessary_columns(df):
    """Remove non-predictive columns"""
    print("Dropping unnecessary columns...")
    columns_to_drop = ['Loan_ID', 'ID', 'id', 'index']
    dropped = [col for col in columns_to_drop if col in df.columns]
    if dropped:
        df.drop(columns=dropped, inplace=True)
        print(f"  ✓ Dropped: {dropped}")
    return df


def rename_target_column(df, target_column):
    """Rename target to standard 'Loan_Status' if different"""
    if target_column != 'Loan_Status':
        df.rename(columns={target_column: 'Loan_Status'}, inplace=True)
        print(f"✓ Renamed '{target_column}' to 'Loan_Status'")
    return df


def preprocess_data(filepath):
    """Complete preprocessing pipeline"""
    print("\n" + "=" * 60)
    print("📥 DATA PREPROCESSING")
    print("=" * 60)

    df = load_data(filepath)
    print(f"\n📋 Initial columns: {df.columns.tolist()}")

    # ✅ CLEAN COLUMN NAMES (remove spaces)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    print(f"📋 Cleaned columns: {df.columns.tolist()}")

    # Find target column
    target_col = find_target_column(df)

    # Preprocessing steps
    df = handle_missing_values(df)
    df = encode_categorical(df, target_col)
    df = drop_unnecessary_columns(df)
    df = rename_target_column(df, target_col)

    print(f"\n✓ Final columns: {df.columns.tolist()}")
    print(f"✓ Data preprocessing complete! Shape: {df.shape}")
    print("=" * 60)

    return df
