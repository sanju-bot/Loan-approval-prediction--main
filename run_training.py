import os
import sys

# Navigate to the correct directory
os.chdir(r'D:\Loan-approval-prediction--main\Loan-approval-prediction--main')

# Delete old model files if they exist
print("Checking for old model files...")
if os.path.exists('best_model.pkl'):
    os.remove('best_model.pkl')
    print("✓ Deleted best_model.pkl")
else:
    print("✗ best_model.pkl not found (already cleaned)")

if os.path.exists('scaler.pkl'):
    os.remove('scaler.pkl')
    print("✓ Deleted scaler.pkl")
else:
    print("✗ scaler.pkl not found (already cleaned)")

print("\n" + "="*60)
print("Running training script: train_model_simple.py")
print("="*60 + "\n")

# Run the training script
os.system('python train_model_simple.py')
