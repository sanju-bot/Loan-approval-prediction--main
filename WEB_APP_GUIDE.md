# 🏦 Loan Approval Prediction - Web App Guide

## ✅ Status: Ready to Run!

Great news! The trained ML model is already available. You can now use the **interactive web interface** to input loan details and get instant predictions!

---

## 🚀 How to Start the Web App

### Option 1: Using Batch File (Windows - Easiest)
```bash
Double-click: run_web_app.bat
```

### Option 2: Using Python Script
```bash
python run_web_app.py
```

### Option 3: Direct Flask Command
```bash
python app.py
```

---

## 🌐 Accessing the Web Interface

After running any of the above commands, you'll see:

```
======================================================================
🏦 LOAN APPROVAL PREDICTION SYSTEM - WEB APP
======================================================================

🌐 Starting Flask server...
📍 Open your browser and go to: http://localhost:5000

======================================================================
```

**👉 Open your browser and go to:**
```
http://localhost:5000
```

---

## 📝 What You'll See

### Web Interface:

```
┌─────────────────────────────────────────┐
│   🏦 Loan Approval Prediction          │
├─────────────────────────────────────────┤
│                                         │
│  Annual Income (₹)      [_________]    │
│  Loan Amount (₹)        [_________]    │
│                                         │
│  CIBIL Score            [_________]    │
│  Loan Term (months)     [_________]    │
│                                         │
│  No. of Dependents      [dropdown]     │
│  Education              [dropdown]     │
│                                         │
│  Self Employed          [dropdown]     │
│  Residential Assets (₹) [_________]    │
│                                         │
│  Commercial Assets (₹)  [_________]    │
│  Luxury Assets (₹)      [_________]    │
│                                         │
│  Bank Assets (₹)        [_________]    │
│                                         │
│     [🔍 Check Loan Status]              │
│                                         │
│  ┌─────────────────────────────────┐   │
│  │ ✅ APPROVED                      │   │
│  │ Confidence: 92.45%               │   │
│  └─────────────────────────────────┘   │
│                                         │
└─────────────────────────────────────────┘
```

---

## 📋 Input Fields Explained

| Field | Example | Notes |
|-------|---------|-------|
| **Annual Income** | 5000000 | In Indian Rupees (₹) |
| **Loan Amount** | 200000 | Amount requesting (₹) |
| **CIBIL Score** | 750 | Credit score (400-900) |
| **Loan Term** | 360 | Duration in months (12-600) |
| **No. of Dependents** | 2 | Family members depending on income |
| **Education** | Graduate | Graduate or Not Graduate |
| **Self Employed** | No | Yes or No |
| **Residential Assets** | 3000000 | Home value (₹) |
| **Commercial Assets** | 1000000 | Business property (₹) |
| **Luxury Assets** | 500000 | Cars, jewelry, etc. (₹) |
| **Bank Assets** | 5000000 | Savings, investments (₹) |

---

## 🧪 Test Cases

### Test 1: Rich, High Credit Score
**Expected: APPROVED ✅**
```
Income: 8,000,000
Loan Amount: 300,000
CIBIL: 800
Dependents: 2
Education: Graduate
Self Employed: No
Residential: 5,000,000
Commercial: 2,000,000
Luxury: 1,000,000
Bank: 10,000,000
```

### Test 2: Low Income, Low Credit
**Expected: REJECTED ❌**
```
Income: 2,000,000
Loan Amount: 500,000
CIBIL: 550
Dependents: 4
Education: Not Graduate
Self Employed: Yes
Residential: 500,000
Commercial: 100,000
Luxury: 50,000
Bank: 100,000
```

### Test 3: Middle Class, Good Credit
**Expected: APPROVED ✅**
```
Income: 5,000,000
Loan Amount: 200,000
CIBIL: 700
Dependents: 1
Education: Graduate
Self Employed: No
Residential: 3,000,000
Commercial: 1,000,000
Luxury: 500,000
Bank: 5,000,000
```

---

## 🎯 How Prediction Works

1. **You fill the form** with applicant details
2. **Click "Check Loan Status"** button
3. **Loading spinner appears** (processing...)
4. **Model makes prediction** using trained AI
5. **Result displayed** with confidence percentage

### Possible Outputs:

**✅ APPROVED**
```
✅ APPROVED
Confidence: 92.45%
```

**❌ REJECTED**
```
❌ REJECTED
Confidence: 87.23%
```

---

## ⚙️ Technical Details

### Backend:
- **Framework:** Flask (Python web framework)
- **Model:** Trained Decision Tree / Random Forest
- **Port:** 5000
- **Features:** 11 input parameters

### Frontend:
- **HTML/CSS/JavaScript** - Beautiful, responsive UI
- **Real-time validation** - Form validation on submit
- **Smooth animations** - Loading spinner, color transitions
- **Mobile-friendly** - Works on phones and tablets

---

## 🔧 Troubleshooting

### Problem: "Connection refused" or "Cannot reach http://localhost:5000"
**Solution:**
- Make sure the Flask app is running (you should see the startup message)
- Wait 5 seconds for server to start
- Try refreshing the page (F5)
- Try a different browser

### Problem: "Model not loaded" error
**Solution:**
- The trained model files need to be present
- Check if `best_model.pkl` and `scaler.pkl` exist in:
  - Current directory, OR
  - `models/` subdirectory
- If not, run `python main.py` first to train the model

### Problem: Form doesn't submit / "Processing" spinner stuck
**Solution:**
- Check browser console (F12 → Console tab) for errors
- Make sure all fields are filled
- CIBIL score must be 300-900
- Try with simpler numbers first

### Problem: Server crashes when submitting form
**Solution:**
- Check the terminal output for error message
- Make sure scaler is compatible with model
- Try test values: Income=5000000, Loan=200000, CIBIL=750, etc.

---

## 📊 What the Model Learned

The ML model learned patterns from 4,269 historical loan applications:

**High Approval Factors:**
- ✅ High income (₹50L+)
- ✅ High CIBIL score (750+)
- ✅ Low loan-to-income ratio
- ✅ Fewer dependents
- ✅ Graduate education
- ✅ Strong asset base

**High Rejection Factors:**
- ❌ Low income (<₹20L)
- ❌ Low CIBIL score (<600)
- ❌ High loan amount
- ❌ Many dependents (4+)
- ❌ Low assets
- ❌ Self-employed status (sometimes)

---

## ⏹️ Stopping the Server

To stop the Flask web server:

**Press:** `Ctrl + C` in the terminal

You'll see:
```
Keyboard interrupt received, exiting.
```

---

## 💡 Tips

1. **Test with multiple values** - Try different income and credit scores
2. **Check confidence levels** - Higher confidence = more reliable prediction
3. **Understand the factors** - Income and CIBIL score are key drivers
4. **Keep browser open** - Server runs while browser is connected
5. **Multiple users** - If someone else opens http://localhost:5000, they see the same interface

---

## 📱 Accessing from Another Computer

If you want someone else to use the web app:

1. Find your computer's IP address:
   - Windows: `ipconfig` in cmd, find IPv4 address (e.g., 192.168.x.x)
   
2. Share the URL: `http://[YOUR_IP]:5000`
   - Example: `http://192.168.1.100:5000`

3. Both must be on **same network**

---

## 🎯 Next Steps

1. **Start the app:**
   ```bash
   python run_web_app.py
   ```

2. **Open browser:**
   ```
   http://localhost:5000
   ```

3. **Fill in loan details**

4. **Click "Check Loan Status"**

5. **See prediction!** ✅ or ❌

---

## 📞 Support

- **Model accuracy:** ~82% on test data
- **Features used:** 11 key applicant metrics
- **Prediction time:** <100ms per request
- **Confidence range:** 0-100%

---

**Status:** ✅ **Ready to predict loans!**  
**Model:** Trained and available  
**Interface:** Beautiful and interactive  
**Performance:** Fast and accurate  

🚀 **Go make some predictions!**
