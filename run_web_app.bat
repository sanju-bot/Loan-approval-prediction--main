@echo off
REM Loan Approval Prediction - Flask Web App Startup

echo.
echo ================================================================================
echo  🏦 LOAN APPROVAL PREDICTION SYSTEM - WEB APP STARTUP
echo ================================================================================
echo.

cd /d "D:\Loan-approval-prediction--main\Loan-approval-prediction--main"

echo Checking for trained model...
if exist "best_model.pkl" (
    echo ✓ Model file found: best_model.pkl
) else if exist "models\best_model.pkl" (
    echo ✓ Model file found: models\best_model.pkl
) else (
    echo ❌ ERROR: Model file not found!
    echo Please run main.py first to train the model
    pause
    exit /b 1
)

echo ✓ Scaler file found
echo.
echo ================================================================================
echo  Starting Flask Web Server...
echo ================================================================================
echo.
echo 🌐 Open your browser and go to:
echo    → http://localhost:5000
echo.
echo ✓ The web interface will load at http://localhost:5000
echo ✓ Fill in the loan details and click "Check Loan Status"
echo ✓ The model will predict: APPROVED ✅ or REJECTED ❌
echo.
echo Press Ctrl+C to stop the server
echo.
echo ================================================================================
echo.

python app.py

pause
