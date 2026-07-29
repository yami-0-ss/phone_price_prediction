import os
import joblib
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load the trained model
MODEL_PATH = "AdaBoost_model.pkl"

try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model file not found or failed to load."}), 500

    try:
        # Extract features from form input
        # Note: Ensure categorical inputs are encoded as expected by your model
        data = [
            float(request.form.get("brand", 0)),
            float(request.form.get("model", 0)),
            float(request.form.get("release_year", 2022)),
            float(request.form.get("ram_gb", 8)),
            float(request.form.get("storage_gb", 128)),
            float(request.form.get("screen_size_inches", 6.1)),
            float(request.form.get("battery_capacity", 4000)),
            float(request.form.get("processor_score", 80)),
            float(request.form.get("camera_score", 80)),
            float(request.form.get("os_type", 0)),
            float(request.form.get("has_5g", 1)),
            float(request.form.get("original_price", 500)),
            float(request.form.get("purchase_year", 2022)),
            float(request.form.get("age_months", 24)),
            float(request.form.get("usage_hours_per_day", 5)),
            float(request.form.get("condition", 1)),
            float(request.form.get("battery_health", 85)),
            float(request.form.get("screen_cracked", 0)),
            float(request.form.get("body_damage", 0)),
            float(request.form.get("repair_history", 0)),
            float(request.form.get("water_damage", 0)),
            float(request.form.get("city_tier", 1)),
            float(request.form.get("seller_type", 0)),
            float(request.form.get("warranty_remaining_months", 0)),
            float(request.form.get("box_available", 1)),
            float(request.form.get("charger_available", 1)),
            float(request.form.get("market_demand_score", 70))
        ]

        features = np.array([data])
        prediction = model.predict(features)[0]

        return jsonify({
            "status": "success",
            "prediction": round(float(prediction), 2)
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
