import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def create_advanced_features(df):
    """Create derived features for better predictions"""
    print("\n🔧 Creating Advanced Features...")

    df_copy = df.copy()

    # 1. Debt-to-Income Ratio
    df_copy['debt_to_income_ratio'] = df_copy['loan_amount'] / df_copy['income_annum']
    print("  ✓ Created: debt_to_income_ratio")

    # 2. Total Assets
    asset_cols = ['residential_assets_value', 'commercial_assets_value',
                  'luxury_assets_value', 'bank_asset_value']
    df_copy['total_assets'] = df_copy[asset_cols].sum(axis=1)
    print("  ✓ Created: total_assets")

    # 3. Assets-to-Loan Ratio
    df_copy['assets_to_loan_ratio'] = df_copy['total_assets'] / df_copy['loan_amount']
    print("  ✓ Created: assets_to_loan_ratio")

    # 4. Has Luxury Assets
    df_copy['has_luxury_assets'] = (df_copy['luxury_assets_value'] > 0).astype(int)
    print("  ✓ Created: has_luxury_assets")

    # 5. Monthly Income
    df_copy['monthly_income'] = df_copy['income_annum'] / 12
    print("  ✓ Created: monthly_income")

    # 6. Monthly EMI
    df_copy['monthly_emi'] = df_copy['loan_amount'] / (df_copy['loan_term'] / 12)
    print("  ✓ Created: monthly_emi")

    # 7. EMI to Income Ratio
    df_copy['emi_to_income_ratio'] = df_copy['monthly_emi'] / df_copy['monthly_income']
    print("  ✓ Created: emi_to_income_ratio")

    # 8. Credit Score Category
    df_copy['credit_score_category'] = pd.cut(
        df_copy['cibil_score'],
        bins=[0, 550, 650, 750, 900],
        labels=[0, 1, 2, 3]
    )
    print("  ✓ Created: credit_score_category")

    # 9. Income Category
    df_copy['income_category'] = pd.qcut(
        df_copy['income_annum'],
        q=4,
        labels=[0, 1, 2, 3],
        duplicates='drop'
    )
    print("  ✓ Created: income_category")

    # 10. Residential Asset Ratio
    df_copy['residential_asset_ratio'] = (df_copy['residential_assets_value'] /
                                          df_copy['total_assets']).fillna(0)
    print("  ✓ Created: residential_asset_ratio")

    # 11. Log transformations for skewed features
    df_copy['log_income'] = np.log1p(df_copy['income_annum'])
    df_copy['log_loan_amount'] = np.log1p(df_copy['loan_amount'])
    df_copy['log_total_assets'] = np.log1p(df_copy['total_assets'])
    print("  ✓ Created: log transformations")

    print(f"\n✓ Advanced Features Created! New shape: {df_copy.shape}")
    print(f"New features: {df_copy.columns.difference(df.columns).tolist()}")

    return df_copy


def handle_outliers(df, columns=None, method='iqr', threshold=1.5):
    """Handle outliers using IQR or Z-score"""
    print("\n🎯 Handling Outliers...")

    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns

    df_copy = df.copy()

    if method == 'iqr':
        for col in columns:
            Q1 = df_copy[col].quantile(0.25)
            Q3 = df_copy[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR

            outliers = df_copy[(df_copy[col] < lower_bound) | (df_copy[col] > upper_bound)]

            # Cap outliers
            df_copy[col] = df_copy[col].clip(lower_bound, upper_bound)

            print(f"  ✓ {col}: {len(outliers)} outliers capped")

    return df_copy