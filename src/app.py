from flask import Flask, request, jsonify
import joblib
import pandas as pd
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# Path to the trained model
MODEL_PATH = 'models/model.joblib'

# Load the model
try:
    model = joblib.load(MODEL_PATH)
    logging.info("Model loaded successfully.")
except FileNotFoundError:
    logging.error(f"Model file not found at {MODEL_PATH}. Please train the model first.")
    model = None
except Exception as e:
    logging.error(f"Error loading model: {e}")
    model = None

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint."""
    logging.info("Health check requested.")
    return jsonify({'status': 'healthy', 'model_loaded': model is not None}), 200

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint."""
    if model is None:
        logging.error("Prediction requested but model is not loaded.")
        return jsonify({'error': 'Model not loaded. Please ensure the model is trained and available.'}), 500

    data = request.get_json(force=True)
    logging.info(f"Received prediction request with data: {data}")

    if not data or not isinstance(data, dict):
        logging.warning("Invalid input: No JSON data or not a dictionary.")
        return jsonify({'error': 'Invalid JSON input. Please send a dictionary of features.'}), 400

    try:
        # Convert input data to DataFrame, ensuring feature order matches training
        # For simplicity, assuming input directly matches model's expected features
        # In a real-world scenario, you'd want a more robust feature engineering pipeline
        input_df = pd.DataFrame([data])

        # Ensure all expected features are present and in the correct order
        # This requires knowing the features the model was trained on. For now, we'll assume
        # the input JSON keys match the column names expected by the model.
        # A better approach would be to save the feature columns from training and use them here.
        # For this exercise, we'll get the features from the trained model's expected features if available.
        # If not, we'll assume the input_df columns are correct.
        if hasattr(model, 'feature_names_in_'):
            expected_features = model.feature_names_in_ # scikit-learn >= 0.23
            # Reorder and potentially add missing features with default values (e.g., 0 or mean)
            missing_features = set(expected_features) - set(input_df.columns)
            for feature in missing_features:
                input_df[feature] = 0 # Or a more appropriate default/imputation strategy
            input_df = input_df[expected_features]
        else:
            # Fallback for older scikit-learn versions or if feature_names_in_ is not available
            # This is less robust and relies on the client sending all features in correct order
            logging.warning("model.feature_names_in_ not found. Relying on input JSON for feature order.")
            
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)

        result = {
            'prediction': int(prediction[0]), # Convert numpy int to Python int
            'probability_benign': float(prediction_proba[0][0]), # Class 0 (Benign)
            'probability_malignant': float(prediction_proba[0][1]) # Class 1 (Malignant)
        }
        logging.info(f"Prediction successful: {result}")
        return jsonify(result), 200

    except KeyError as e:
        logging.error(f"Missing feature in input data: {e}")
        return jsonify({'error': f'Missing feature in input data: {e}. Please ensure all required features are provided.'}), 400
    except Exception as e:
        logging.error(f"Error during prediction: {e}", exc_info=True)
        return jsonify({'error': f'An internal error occurred: {e}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
