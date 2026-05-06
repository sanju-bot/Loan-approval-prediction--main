#!/usr/bin/env python
"""
Standalone execution script that runs main.py and logs all output
"""
import subprocess
import sys
import os

# Change to script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

print("=" * 80)
print("🏦 LOAN APPROVAL PREDICTION - EXECUTION START")
print("=" * 80)
print(f"Working Directory: {os.getcwd()}")
print(f"Python Version: {sys.version}")
print("=" * 80 + "\n")

# Run main.py
try:
    result = subprocess.run(
        [sys.executable, 'main.py'],
        capture_output=False,
        text=True,
        timeout=600  # 10 minute timeout
    )
    sys.exit(result.returncode)
except subprocess.TimeoutExpired:
    print("\n❌ ERROR: Execution timeout after 10 minutes")
    sys.exit(1)
except Exception as e:
    print(f"\n❌ ERROR: {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
