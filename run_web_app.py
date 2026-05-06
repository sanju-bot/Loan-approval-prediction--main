#!/usr/bin/env python
"""
Loan Approval Prediction - Flask Web App Launcher
Run this to start the interactive web interface
"""
import os
import sys
import subprocess
from pathlib import Path

# Get script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

print("\n" + "=" * 80)
print("🏦 LOAN APPROVAL PREDICTION SYSTEM - WEB APP LAUNCHER")
print("=" * 80)

# Check for model files
model_paths = [
    'best_model.pkl',
    'models/best_model.pkl'
]
scaler_paths = [
    'scaler.pkl',
    'models/scaler.pkl'
]

model_exists = any(Path(p).exists() for p in model_paths)
scaler_exists = any(Path(p).exists() for p in scaler_paths)

if not model_exists or not scaler_exists:
    print("\n❌ ERROR: Trained model files not found!")
    print("\nRequired files:")
    print("  - best_model.pkl (or models/best_model.pkl)")
    print("  - scaler.pkl (or models/scaler.pkl)")
    print("\n⚠️  Please run 'python main.py' first to train the model")
    print("\nSteps:")
    print("  1. Run: python main.py")
    print("  2. Wait for training to complete")
    print("  3. Then run: python run_web_app.py")
    input("\n👉 Press ENTER to exit...")
    sys.exit(1)

print("\n✓ Model files found!")
print("✓ Scaler file found!")

print("\n" + "=" * 80)
print("🌐 Starting Flask Web Server...")
print("=" * 80)
print("\n📱 Web Interface Details:")
print("  → URL: http://localhost:5000")
print("  → Open this link in your web browser")
print("\n🎯 How to use:")
print("  1. Fill in the loan applicant details")
print("  2. Click '🔍 Check Loan Status' button")
print("  3. The model will predict: ✅ APPROVED or ❌ REJECTED")
print("  4. See the confidence percentage")
print("\n⏹️  To stop the server: Press Ctrl+C")
print("\n" + "=" * 80 + "\n")

# Launch Flask app
try:
    subprocess.run([sys.executable, 'app.py'], check=False)
except KeyboardInterrupt:
    print("\n\n🛑 Server stopped by user")
    sys.exit(0)
except Exception as e:
    print(f"\n❌ Error: {e}")
    sys.exit(1)
