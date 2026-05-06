from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import os
import joblib
import subprocess
import sys

app = Flask(__name__)

# Load trained model and scaler
model = None
scaler = None

def try_load_model():
    """Try to load model, if fails, train it"""
    global model, scaler
    
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Try to load existing model
        model_candidates = [
            'best_model.pkl',
            os.path.join(base_dir, 'best_model.pkl'),
            os.path.join(base_dir, 'models', 'best_model.pkl')
        ]
        scaler_candidates = [
            'scaler.pkl',
            os.path.join(base_dir, 'scaler.pkl'),
            os.path.join(base_dir, 'models', 'scaler.pkl')
        ]
        
        model_path = next((p for p in model_candidates if os.path.exists(p)), None)
        scaler_path = next((p for p in scaler_candidates if os.path.exists(p)), None)
        
        if model_path and scaler_path:
            print(f"📥 Loading model from: {model_path}")
            model = joblib.load(model_path)
            print(f"✓ Model loaded successfully!")
            
            print(f"📥 Loading scaler from: {scaler_path}")
            scaler = joblib.load(scaler_path)
            print(f"✓ Scaler loaded successfully!")
            return True
        else:
            print("❌ Model or scaler files not found")
            return False
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

# Try to load model on startup
print("\n" + "=" * 70)
print("🏦 LOAN APPROVAL PREDICTION SYSTEM - WEB APP")
print("=" * 70 + "\n")

if not try_load_model():
    print("\n⚠️  Model files not found. Auto-training model...")
    print("This will take 1-2 minutes...\n")
    
    try:
        result = subprocess.run(
            [sys.executable, 'train_model_simple.py'],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode == 0:
            print(result.stdout)
            print("\n✓ Model trained successfully!")
            print("Now loading model...\n")
            if not try_load_model():
                print("❌ Failed to load model after training!")
        else:
            print(f"❌ Training failed:\n{result.stderr}")
    except Exception as e:
        print(f"❌ Error during auto-training: {e}")

if model is None or scaler is None:
    print("⚠️  WARNING: Model not loaded. Web app will run but predictions will fail.")
    print("Please run: python train_model_simple.py\n")


def predict_from_model(model, scaler, data):
    """Make prediction"""
    prediction = model.predict(data)
    probability = model.predict_proba(data)
    
    result = "✅ APPROVED" if prediction[0] == 1 else "❌ REJECTED"
    confidence = max(probability[0]) * 100
    
    return result, confidence


@app.route('/')
def index():
    """Serve the web interface"""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>🏦 Loan Approval Prediction</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }

            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 20px;
            }

            .container {
                background: white;
                border-radius: 15px;
                box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
                max-width: 600px;
                width: 100%;
                padding: 40px;
            }

            h1 {
                text-align: center;
                color: #333;
                margin-bottom: 30px;
                font-size: 28px;
            }

            .form-group {
                margin-bottom: 20px;
            }

            label {
                display: block;
                margin-bottom: 8px;
                color: #555;
                font-weight: 600;
            }

            input, select {
                width: 100%;
                padding: 12px;
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                font-size: 14px;
                transition: border-color 0.3s;
            }

            input:focus, select:focus {
                outline: none;
                border-color: #667eea;
            }

            .form-row {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 15px;
            }

            button {
                width: 100%;
                padding: 14px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 16px;
                font-weight: 600;
                cursor: pointer;
                margin-top: 20px;
                transition: transform 0.2s, box-shadow 0.2s;
            }

            button:hover {
                transform: translateY(-2px);
                box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4);
            }

            .result {
                margin-top: 30px;
                padding: 20px;
                border-radius: 8px;
                text-align: center;
                display: none;
            }

            .result.approved {
                background: #d4edda;
                border: 2px solid #28a745;
                color: #155724;
            }

            .result.rejected {
                background: #f8d7da;
                border: 2px solid #dc3545;
                color: #721c24;
            }

            .result h2 {
                font-size: 24px;
                margin-bottom: 10px;
            }

            .result p {
                font-size: 18px;
            }

            .loading {
                text-align: center;
                margin: 20px 0;
                display: none;
            }

            .spinner {
                border: 4px solid #f3f3f3;
                border-top: 4px solid #667eea;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
                margin: 0 auto;
            }

            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🏦 Loan Approval Prediction</h1>

            <form id="predictionForm">
                <div class="form-row">
                    <div class="form-group">
                        <label>Annual Income (₹)</label>
                        <input type="number" id="income_annum" required>
                    </div>
                    <div class="form-group">
                        <label>Loan Amount (₹)</label>
                        <input type="number" id="loan_amount" required>
                    </div>
                </div>

                <div class="form-row">
                    <div class="form-group">
                        <label>CIBIL Score</label>
                        <input type="number" id="cibil_score" min="300" max="900" required>
                    </div>
                    <div class="form-group">
                        <label>Loan Term (months)</label>
                        <input type="number" id="loan_term" required>
                    </div>
                </div>

                <div class="form-row">
                    <div class="form-group">
                        <label>No. of Dependents</label>
                        <select id="no_of_dependents" required>
                            <option value="">Select</option>
                            <option value="0">0</option>
                            <option value="1">1</option>
                            <option value="2">2</option>
                            <option value="3">3</option>
                            <option value="4">4</option>
                            <option value="5">5+</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Education</label>
                        <select id="education" required>
                            <option value="">Select</option>
                            <option value="1">Graduate</option>
                            <option value="0">Not Graduate</option>
                        </select>
                    </div>
                </div>

                <div class="form-row">
                    <div class="form-group">
                        <label>Self Employed</label>
                        <select id="self_employed" required>
                            <option value="">Select</option>
                            <option value="0">No</option>
                            <option value="1">Yes</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label>Residential Assets (₹)</label>
                        <input type="number" id="residential_assets_value" required>
                    </div>
                </div>

                <div class="form-row">
                    <div class="form-group">
                        <label>Commercial Assets (₹)</label>
                        <input type="number" id="commercial_assets_value" required>
                    </div>
                    <div class="form-group">
                        <label>Luxury Assets (₹)</label>
                        <input type="number" id="luxury_assets_value" required>
                    </div>
                </div>

                <div class="form-group">
                    <label>Bank Assets (₹)</label>
                    <input type="number" id="bank_asset_value" required>
                </div>

                <button type="submit">🔍 Check Loan Status</button>
            </form>

            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Processing your request...</p>
            </div>

            <div class="result" id="result">
                <h2 id="resultText"></h2>
                <p id="confidenceText"></p>
            </div>
        </div>

        <script>
            document.getElementById('predictionForm').addEventListener('submit', async (e) => {
                e.preventDefault();

                const loading = document.getElementById('loading');
                const result = document.getElementById('result');

                loading.style.display = 'block';
                result.style.display = 'none';

                const data = {
                    no_of_dependents: document.getElementById('no_of_dependents').value,
                    education: document.getElementById('education').value,
                    self_employed: document.getElementById('self_employed').value,
                    income_annum: document.getElementById('income_annum').value,
                    loan_amount: document.getElementById('loan_amount').value,
                    loan_term: document.getElementById('loan_term').value,
                    cibil_score: document.getElementById('cibil_score').value,
                    residential_assets_value: document.getElementById('residential_assets_value').value,
                    commercial_assets_value: document.getElementById('commercial_assets_value').value,
                    luxury_assets_value: document.getElementById('luxury_assets_value').value,
                    bank_asset_value: document.getElementById('bank_asset_value').value
                };

                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify(data)
                    });

                    const responseData = await response.json();

                    loading.style.display = 'none';

                    if (!response.ok || responseData.error) {
                        result.style.display = 'none';
                        throw new Error(responseData.error || `Request failed with status ${response.status}`);
                    }

                    result.style.display = 'block';

                    const isApproved = responseData.code === 1;
                    result.className = `result ${isApproved ? 'approved' : 'rejected'}`;
                    document.getElementById('resultText').textContent = responseData.prediction || (isApproved ? '✅ APPROVED' : '❌ REJECTED');
                    document.getElementById('confidenceText').textContent = `Confidence: ${responseData.confidence ?? 'N/A'}`;
                } catch (error) {
                    loading.style.display = 'none';
                    alert('Error: ' + error.message);
                }
            });
        </script>
    </body>
    </html>
    """
    return html_content


@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        if model is None or scaler is None:
            return jsonify({'error': 'Model not loaded. Please ensure model files exist.'}), 500

        data = request.json

        required_fields = [
            'no_of_dependents', 'education', 'self_employed',
            'income_annum', 'loan_amount', 'loan_term',
            'cibil_score', 'residential_assets_value',
            'commercial_assets_value', 'luxury_assets_value',
            'bank_asset_value'
        ]

        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400

        # Prepare data
        applicant_data = pd.DataFrame({
            'no_of_dependents': [int(data['no_of_dependents'])],
            'education': [int(data['education'])],
            'self_employed': [int(data['self_employed'])],
            'income_annum': [int(data['income_annum'])],
            'loan_amount': [int(data['loan_amount'])],
            'loan_term': [int(data['loan_term'])],
            'cibil_score': [int(data['cibil_score'])],
            'residential_assets_value': [int(data['residential_assets_value'])],
            'commercial_assets_value': [int(data['commercial_assets_value'])],
            'luxury_assets_value': [int(data['luxury_assets_value'])],
            'bank_asset_value': [int(data['bank_asset_value'])]
        })

        # Scale and predict
        applicant_scaled = scaler.transform(applicant_data)
        result, confidence = predict_from_model(model, scaler, applicant_scaled)

        return jsonify({
            'prediction': result,
            'confidence': f"{confidence:.2f}%",
            'code': 1 if '✅' in result else 0
        })

    except Exception as e:
        print(f"❌ Error in predict: {str(e)}")
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500


if __name__ == '__main__':
    print("\n🌐 Starting Flask server...")
    print("📍 Open your browser and go to: http://localhost:5000")
    print("=" * 70 + "\n")
    app.run(debug=False, port=5000, use_reloader=False)
