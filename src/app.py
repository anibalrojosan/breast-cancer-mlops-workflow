from flask import Flask, request, jsonify
import joblib
import pandas as pd
import logging
import sys 
from pydantic import ValidationError
from src.schemas import PredictRequest


# Configure logging
log_file_path = 'api_logs.log'
logging.basicConfig(
    level=logging.INFO,
    format='*** FLASK API LOG: %(asctime)s - %(levelname)s - %(message)s ***',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler(sys.stdout) # Explicitly use sys.stdout for terminal output
    ]
)

app = Flask(__name__)

# Path to the trained model
MODEL_PATH = 'models/model.joblib'

try:
    model = joblib.load(MODEL_PATH)
    logging.info("Model loaded successfully.")
except FileNotFoundError:
    logging.error(f"Model file not found at {MODEL_PATH}. Train the model first.")
    model = None
except Exception as e:
    logging.error(f"Error loading model: {e}")
    model = None

# Endpoints
@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint."""
    logging.info("Health check requested.")
    return jsonify({'status': 'healthy', 'model_loaded': model is not None}), 200

@app.route('/predict', methods=['POST'])
def predict():
    logging.info("Prediction endpoint hit.")
    if model is None:
        logging.error("Prediction requested but model is not loaded.")
        return jsonify({'error': 'Model not loaded. Please ensure the model is trained and available.'}), 500

    # Parse JSON
    try:
        data = request.get_json(force=True)
    except Exception:
        logging.warning("Invalid JSON body.")
        return jsonify({'error': 'Invalid JSON body.'}), 400

    # Validation using Pydantic
    try:
        payload = PredictRequest.model_validate(data)
    except ValidationError as e:
        logging.warning(f"Validation error: {e}")
        return jsonify({'error': 'Invalid input', 'details': e.errors()}), 422

    # Build DataFrame using aliases to match training feature names
    input_df = pd.DataFrame([payload.model_dump(by_alias=True)])

    # Feature alignment
    if hasattr(model, 'feature_names_in_'):
        expected_features = list(model.feature_names_in_) # scikit-learn feature names
        missing_features = set(expected_features) - set(input_df.columns)
        for feature in missing_features:
            input_df[feature] = 0
        input_df = input_df[expected_features]
    else:
        logging.warning("model.feature_names_in_ not found. Relying on input JSON for feature order.")

    # Inference
    try:
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)
    except Exception as e:
        logging.error(f"Error during prediction: {e}", exc_info=True)
        return jsonify({'error': f'An internal error occurred: {e}'}), 500

    result = {
        'prediction': int(prediction[0]),
        'probability_benign': float(prediction_proba[0][0]),       # Class 0 is benign
        'probability_malignant': float(prediction_proba[0][1]),    # Class 1 is malignant
    }
    logging.info(f"Prediction successful: {result}")
    return jsonify(result), 200

if __name__ == '__main__':
    app.run(debug=False, use_reloader=False, host='0.0.0.0', port=5000)
