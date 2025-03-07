from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained models
irrigation_model = joblib.load("./models/irrigation_model.pkl")
fertiliser_model = joblib.load("./models/fertiliser_model.pkl")

# Define thresholds for nutrient levels
def categorize_nutrient(n, p, k):
    n_status = "Deficient" if n < 50 else "Healthy" if 50 <= n <= 200 else "Excess"
    p_status = "Deficient" if p < 20 else "Healthy" if 20 <= p <= 60 else "Excess"
    k_status = "Deficient" if k < 120 else "Healthy" if 120 <= k <= 250 else "Excess"
    return n_status, p_status, k_status

# Define action recommendations based on nutrient status
action_map = {
    "Deficient": {
        "Nitrogen (N)": "Apply biochar with high nitrogen organic amendments (e.g., manure). Increase cover crops. Use precision irrigation to manage leaching.",
        "Phosphorus (P)": "Apply phosphorus-rich organic amendments (e.g., bone meal) and biochar to improve P availability. Increase soil pH if needed.",
        "Potassium (K)": "Use potassium-rich organic amendments (e.g., compost with banana peels) combined with biochar to boost K levels."
    },
    "Healthy": {
        "Nitrogen (N)": "Maintain current biochar and organic amendment levels. Monitor periodically.",
        "Phosphorus (P)": "Maintain phosphorus application at current levels. Monitor regularly for changes in soil P.",
        "Potassium (K)": "Keep K applications steady and monitor for depletion. Avoid over-amendment."
    },
    "Excess": {
        "Nitrogen (N)": "Reduce nitrogen input. Adjust irrigation to prevent leaching. Consider adding carbon-rich biochar to balance.",
        "Phosphorus (P)": "Stop phosphorus amendments. Apply carbon-rich biochar to lock excess P and prevent runoff.",
        "Potassium (K)": "Reduce potassium inputs. Introduce organic matter to improve soil structure and prevent excess K leaching."
    }
}

# Function to determine water requirement based on soil moisture
def water_requirement(moisture):
    if moisture < 30:
        return "High water requirement: 1-2 liters per plant"
    elif 30 <= moisture < 60:
        return "Medium water requirement: 0.5-1 liters per plant"
    else:
        return "Low water requirement: 0-0.5 liters per plant"

# Irrigation Prediction API
@app.route("/predict_irrigation", methods=["POST"])
def predict_irrigation():
    data = request.json
    input_data = pd.DataFrame([[data["moisture"], data["temperature"], data["humidity"]]], 
                              columns=["moisture", "temperature", "humidity"])
    
    irrigation_needed = irrigation_model.predict(input_data)[0]
    water_suggestion = water_requirement(data["moisture"])
    
    return jsonify({
        "Irrigation Required": bool(irrigation_needed),
        "Water Suggestion": water_suggestion
    })

# Fertilizer Prediction API (Now Includes Actions)
@app.route("/predict_fertiliser", methods=["POST"])
def predict_fertiliser():
    data = request.json
    nitrogen, phosphorus, potassium = data["nitrogen"], data["phosphorus"], data["potassium"]

    input_data = pd.DataFrame([[nitrogen, phosphorus, potassium]], 
                              columns=["Nitrogen (N)", "Phosphorus (P)", "Potassium (K)"])
    
    prediction = fertiliser_model.predict(input_data)[0]
    fertilizer_status = bool(prediction)

    # Categorize nutrient levels
    n_status, p_status, k_status = categorize_nutrient(nitrogen, phosphorus, potassium)

    # Get recommended actions
    n_action = action_map[n_status]["Nitrogen (N)"]
    p_action = action_map[p_status]["Phosphorus (P)"]
    k_action = action_map[k_status]["Potassium (K)"]

    return jsonify({
        "Fertilizer Required": fertilizer_status,
        "Nitrogen Status": n_status,
        "Phosphorus Status": p_status,
        "Potassium Status": k_status,
        "Nitrogen Recommendation": n_action,
        "Phosphorus Recommendation": p_action,
        "Potassium Recommendation": k_action
    })

# Default route
@app.route("/")
def home():
    return "Smart Irrigation & Fertilization API is running!"

if __name__ == "__main__":
    app.run(debug=True)
