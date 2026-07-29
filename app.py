import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# List of all 27 features stored in your model
FEATURE_NAMES = [
    "brand", "model", "release_year", "ram_gb", "storage_gb",
    "screen_size_inches", "battery_capacity", "processor_score", "camera_score",
    "os_type", "has_5g", "original_price", "purchase_year", "age_months",
    "usage_hours_per_day", "condition", "battery_health", "screen_cracked",
    "body_damage", "repair_history", "water_damage", "city_tier",
    "seller_type", "warranty_remaining_months", "box_available",
    "charger_available", "market_demand_score"
]

# Check case-sensitive file path for Render
MODEL_PATH = "Adaboost_model.pkl" if os.path.exists("Adaboost_model.pkl") else "AdaBoost_model.pkl"

model = None
if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        print(f"Successfully loaded model from {MODEL_PATH}")
    except Exception as e:
        print(f"Error unpickling model file: {e}")
else:
    print(f"Warning: {MODEL_PATH} not found.")

HTML_LAYOUT = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Device Price Estimator ✦</title>
  <link href="https://fonts.googleapis.com/css2?family=Fredoka:wght@400;600;700&family=Nunito:wght@400;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-gradient: linear-gradient(135deg, #a8edd9 0%, #fed6e3 100%);
      --card-bg: rgba(255, 255, 255, 0.92);
      --primary: #ff7675;
      --primary-hover: #e84393;
      --text: #2d3436;
      --radius: 20px;
    }

    * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Nunito', sans-serif; }

    body {
      background: var(--bg-gradient);
      min-height: 100vh;
      display: flex;
      justify-content: center;
      align-items: center;
      padding: 20px;
      position: relative;
      overflow-x: hidden;
    }

    /* Floating shapes */
    .shape {
      position: absolute;
      border-radius: 50%;
      filter: blur(40px);
      z-index: 0;
      animation: float 8s ease-in-out infinite alternate;
    }
    .shape-1 { width: 250px; height: 250px; background: #fd79a8; top: 10%; left: 10%; }
    .shape-2 { width: 300px; height: 300px; background: #a29bfe; bottom: 10%; right: 10%; animation-delay: -4s; }

    @keyframes float {
      0% { transform: translateY(0px) rotate(0deg); }
      100% { transform: translateY(30px) rotate(15deg); }
    }

    .container {
      background: var(--card-bg);
      backdrop-filter: blur(10px);
      width: 100%;
      max-width: 800px;
      border-radius: var(--radius);
      padding: 35px;
      box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
      z-index: 1;
      position: relative;
    }

    .header { text-align: center; margin-bottom: 25px; }

    .mascot {
      width: 80px; height: 80px;
      margin: 0 auto 10px;
      background: #ffeaa7;
      border-radius: 50%;
      display: flex; align-items: center; justify-content: center;
      font-size: 40px;
      box-shadow: 0 8px 15px rgba(0, 0, 0, 0.08);
      animation: bounce 2s infinite;
    }

    @keyframes bounce {
      0%, 100% { transform: translateY(0); }
      50% { transform: translateY(-10px); }
    }

    h1 { font-family: 'Fredoka', sans-serif; color: var(--text); font-size: 28px; }
    p.subtitle { color: #636e72; font-size: 15px; margin-top: 5px; }

    .form-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 15px;
    }

    .input-group { display: flex; flex-direction: column; }
    .input-group label { font-size: 13px; font-weight: 700; color: #2d3436; margin-bottom: 5px; }
    .input-group input, .input-group select {
      padding: 10px 14px;
      border: 2px solid #dfe6e9;
      border-radius: 12px;
      outline: none;
      font-size: 14px;
      transition: all 0.3s ease;
      background: #fdfdfd;
    }
    .input-group input:focus, .input-group select:focus {
      border-color: var(--primary);
      box-shadow: 0 0 0 4px rgba(255, 118, 117, 0.2);
    }

    .btn-submit {
      grid-column: 1 / -1;
      margin-top: 15px;
      padding: 14px;
      border: none;
      border-radius: 12px;
      background: var(--primary);
      color: white;
      font-size: 16px;
      font-weight: 700;
      cursor: pointer;
      transition: all 0.3s ease;
      box-shadow: 0 8px 15px rgba(255, 118, 117, 0.3);
    }

    .btn-submit:hover {
      background: var(--primary-hover);
      transform: translateY(-2px);
    }

    .result-box {
      margin-top: 25px;
      padding: 20px;
      border-radius: 15px;
      background: #e8f8f5;
      border: 2px dashed #55efc4;
      text-align: center;
      display: none;
      animation: fadeIn 0.5s ease-out forwards;
    }

    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(10px); }
      to { opacity: 1; transform: translateY(0); }
    }

    .result-box h2 { font-family: 'Fredoka', sans-serif; color: #00b894; font-size: 26px; }
    .loader {
      display: none;
      margin: 20px auto;
      border: 4px solid #f3f3f3;
      border-top: 4px solid var(--primary);
      border-radius: 50%;
      width: 35px; height: 35px;
      animation: spin 1s linear infinite;
    }
    @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
  </style>
</head>
<body>
  <div class="shape shape-1"></div>
  <div class="shape shape-2"></div>

  <div class="container">
    <div class="header">
      <div class="mascot">📱✨</div>
      <h1>Device Valuation AI</h1>
      <p class="subtitle">Enter device specifications to estimate value</p>
    </div>

    <form id="prediction-form" class="form-grid">
      <div class="input-group">
        <label>Release Year</label>
        <input type="number" name="release_year" value="2022" required>
      </div>
      <div class="input-group">
        <label>RAM (GB)</label>
        <input type="number" name="ram_gb" value="8" required>
      </div>
      <div class="input-group">
        <label>Storage (GB)</label>
        <input type="number" name="storage_gb" value="128" required>
      </div>
      <div class="input-group">
        <label>Screen Size (inches)</label>
        <input type="number" step="0.1" name="screen_size_inches" value="6.1" required>
      </div>
      <div class="input-group">
        <label>Battery Capacity (mAh)</label>
        <input type="number" name="battery_capacity" value="4000" required>
      </div>
      <div class="input-group">
        <label>Battery Health (%)</label>
        <input type="number" name="battery_health" value="85" required>
      </div>
      <div class="input-group">
        <label>Original Price ($)</label>
        <input type="number" name="original_price" value="600" required>
      </div>
      <div class="input-group">
        <label>Age (Months)</label>
        <input type="number" name="age_months" value="18" required>
      </div>
      <div class="input-group">
        <label>Has 5G?</label>
        <select name="has_5g">
          <option value="1">Yes</option>
          <option value="0">No</option>
        </select>
      </div>
      <div class="input-group">
        <label>Screen Cracked?</label>
        <select name="screen_cracked">
          <option value="0">No</option>
          <option value="1">Yes</option>
        </select>
      </div>

      <button type="submit" class="btn-submit">Calculate Valuation 🚀</button>
    </form>

    <div class="loader" id="loader"></div>

    <div class="result-box" id="result-box">
      <p style="color: #636e72; font-weight: 600;">Estimated Market Value</p>
      <h2 id="prediction-text">$0.00</h2>
    </div>
  </div>

  <script>
    document.getElementById('prediction-form').addEventListener('submit', async function(e) {
      e.preventDefault();
      const loader = document.getElementById('loader');
      const resultBox = document.getElementById('result-box');
      
      resultBox.style.display = 'none';
      loader.style.display = 'block';

      const formData = new FormData(this);

      try {
        const response = await fetch('/predict', { method: 'POST', body: formData });
        const data = await response.json();
        loader.style.display = 'none';

        if (data.status === 'success') {
          document.getElementById('prediction-text').innerText = '$' + data.prediction;
          resultBox.style.display = 'block';
        } else {
          alert('Error: ' + data.message);
        }
      } catch (err) {
        loader.style.display = 'none';
        alert('An unexpected error occurred: ' + err);
      }
    });
  </script>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_LAYOUT)

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({
            "status": "error",
            "message": "Model file missing or failed to unpickle."
        }), 500

    try:
        input_data = {}
        for feature in FEATURE_NAMES:
            val = request.form.get(feature)
            if val is not None and val.strip() != "":
                input_data[feature] = float(val)
            else:
                input_data[feature] = 0.0

        df = pd.DataFrame([input_data])
        prediction = model.predict(df)[0]

        return jsonify({
            "status": "success",
            "prediction": round(float(prediction), 2)
        })

    except Exception as e:
        print(f"Prediction Error: {str(e)}")
        return jsonify({
            "status": "error",
            "message": f"Server Prediction Error: {str(e)}"
        }), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
