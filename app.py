import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# List of all 27 features saved in the model
FEATURE_NAMES = [
    "brand", "model", "release_year", "ram_gb", "storage_gb",
    "screen_size_inches", "battery_capacity", "processor_score", "camera_score",
    "os_type", "has_5g", "original_price", "purchase_year", "age_months",
    "usage_hours_per_day", "condition", "battery_health", "screen_cracked",
    "body_damage", "repair_history", "water_damage", "city_tier",
    "seller_type", "warranty_remaining_months", "box_available",
    "charger_available", "market_demand_score"
]

# Check case-sensitive path for Render (Linux environment)
MODEL_PATH = "Adaboost_model.pkl" if os.path.exists("Adaboost_model.pkl") else "AdaBoost_model.pkl"

model = None
if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        print(f"Model successfully loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"Error unpickling model: {e}")
else:
    print(f"Warning: Model file not found at {MODEL_PATH}")

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({
            "status": "error", 
            "message": "Model file missing or failed to load on the server."
        }), 500

    try:
        # Construct dictionary with fallback values for any unsubmitted fields
        input_data = {}
        for feature in FEATURE_NAMES:
            val = request.form.get(feature)
            if val is not None and val.strip() != "":
                input_data[feature] = float(val)
            else:
                input_data[feature] = 0.0

        # Create DataFrame with expected column order
        df = pd.DataFrame([input_data])
        
        # Predict price/valuation
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
