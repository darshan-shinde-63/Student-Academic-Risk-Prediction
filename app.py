from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import pandas as pd
import os

app = Flask(__name__, static_folder='frontend', static_url_path='')
CORS(app)

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

# Load the trained model pipeline
model_path = 'dropout_model.pkl'
try:
    pipeline = joblib.load(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    pipeline = None

@app.route('/predict', methods=['POST'])
def predict():
    if not pipeline:
        return jsonify({"error": "Model not loaded properly."}), 500
        
    try:
        # Get JSON data
        data = request.get_json()
        
        # Convert to pandas DataFrame (expecting a list of dicts or single dict)
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            return jsonify({"error": "Invalid input format. Expected JSON object or list of objects."}), 400

        # Ensure all columns required are present (the preprocessor will handle extra/missing to some extent, but better to be safe)
        # Note: missing numerical columns will fail StandardScaler, categorical will fail OneHotEncoder if not handled.
        
        # Predict probability
        proba = pipeline.predict_proba(df)[:, 1] # Probability of class 1 (Dropout)
        
        # Determine risk
        results = []
        for p in proba:
            if p < 0.3:
                risk = "Low"
            elif p <= 0.7:
                risk = "Medium"
            else:
                risk = "High"
                
            results.append({
                "probability": float(p),
                "risk": risk
            })
            
        # Return results
        if isinstance(data, dict) and len(results) == 1:
            return jsonify(results[0])
        else:
            return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True, port=5000)
