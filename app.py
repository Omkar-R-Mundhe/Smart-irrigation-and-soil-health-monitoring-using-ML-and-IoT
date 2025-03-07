from flask import Flask, request, jsonify
import pandas as pd
import joblib
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Load trained models
irrigation_model = joblib.load("./models/irrigation_model.pkl")
fertiliser_model = joblib.load("./models/fertiliser_model.pkl")

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

# Fertilizer Prediction API
@app.route("/predict_fertiliser", methods=["POST"])
def predict_fertiliser():
    data = request.json
    input_data = pd.DataFrame([[data["nitrogen"], data["phosphorus"], data["potassium"]]], 
                              columns=["Nitrogen (N)", "Phosphorus (P)", "Potassium (K)"])
    
    prediction = fertiliser_model.predict(input_data)[0]
    
    return jsonify({"Fertilizer Required": bool(prediction)})

# Default route
@app.route("/")
def home():
    return "Smart Irrigation & Fertilization API is running!"

if __name__ == "__main__":
    app.run(debug=True)
