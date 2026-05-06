import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ✅ Use non-interactive backend to prevent blocking
plt.switch_backend('Agg')


def perform_eda(df, output_dir='output/plots/'):
    """Analyze data relationships and patterns"""
    print("Starting Exploratory Data Analysis...")

    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Set style
    sns.set_style("whitegrid")

    # ✅ Check which columns exist
    print(f"Available columns: {df.columns.tolist()}")

    # 1. Loan Status Distribution
    if 'Loan_Status' in df.columns:
        plt.figure(figsize=(8, 5))
        sns.countplot(x='Loan_Status', data=df)
        plt.title('Loan Approval Distribution')
        plt.xlabel('Loan Status (0=Rejected, 1=Approved)')
        plt.savefig(f'{output_dir}loan_distribution.png', dpi=100, bbox_inches='tight')
        plt.close()
        print("  ✓ Saved: loan_distribution.png")
    else:
        print("  ⚠️ 'Loan_Status' column not found!")

    # 2. Income vs Loan Status
    if 'ApplicantIncome' in df.columns and 'Loan_Status' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Loan_Status', y='ApplicantIncome', data=df)
        plt.title('Applicant Income vs Loan Status')
        plt.savefig(f'{output_dir}income_vs_status.png', dpi=100, bbox_inches='tight')
        plt.close()
        print("  ✓ Saved: income_vs_status.png")

    # 3. Credit History Effect
    if 'Credit_History' in df.columns and 'Loan_Status' in df.columns:
        plt.figure(figsize=(8, 5))
        sns.countplot(x='Credit_History', hue='Loan_Status', data=df)
        plt.title('Credit History Impact on Approval')
        plt.savefig(f'{output_dir}credit_history.png', dpi=100, bbox_inches='tight')
        plt.close()
        print("  ✓ Saved: credit_history.png")

    # 4. Correlation Matrix
    plt.figure(figsize=(10, 8))
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig(f'{output_dir}correlation_matrix.png', dpi=100, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: correlation_matrix.png")

    # 5. Missing Values
    plt.figure(figsize=(10, 5))
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        missing.plot(kind='bar')
        plt.title('Missing Values')
        plt.savefig(f'{output_dir}missing_values.png', dpi=100, bbox_inches='tight')
        plt.close()
        print("  ✓ Saved: missing_values.png")

    print("✓ EDA complete! Check output/plots/ folder for visualizations")


def print_statistics(df):
    """Print basic dataset statistics"""
    print("\n" + "=" * 50)
    print("📊 DATASET STATISTICS")
    print("=" * 50)
    print(f"Dataset Shape: {df.shape}")
    print(f"\nColumn Names: {df.columns.tolist()}")
    print(f"\nData Types:\n{df.dtypes}")
    print(f"\nMissing Values:\n{df.isnull().sum()}")
    print(f"\nBasic Statistics:\n{df.describe()}")
    print("=" * 50)