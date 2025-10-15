import joblib
import pandas as pd
import os


def load_pipeline(model_path='models/model.joblib'):
    """Loads the trained scikit-learn pipeline."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model pipeline not found at {model_path}. Please train the model first.")
    return joblib.load(model_path)

def predict(raw_data, model_path='models/model.joblib'):
    """Loads the pipeline and makes a prediction on new raw data."""
    pipeline = load_pipeline(model_path)
    prediction = pipeline.predict(raw_data)
    return prediction

if __name__ == "__main__":
    print("This module is for inference. Please run model_training.py to train the model.")
    try:
        # Example of new raw data (single row DataFrame)
        sample_new_data = pd.DataFrame([{
            'radius_mean': 17.99, 'texture_mean': 10.38,
            'perimeter_mean': 122.8, 'area_mean': 1001.0, 'smoothness_mean': 0.1184,
            'compactness_mean': 0.2776, 'concavity_mean': 0.3001, 'concave points_mean': 0.1471,
            'symmetry_mean': 0.2419, 'fractal_dimension_mean': 0.07871,
            'radius_se': 1.095, 'texture_se': 0.9053, 'perimeter_se': 8.589, 'area_se': 153.4,
            'smoothness_se': 0.006399, 'compactness_se': 0.04904, 'concavity_se': 0.05373,
            'concave points_se': 0.01587, 'symmetry_se': 0.03003, 'fractal_dimension_se': 0.006193,
            'radius_worst': 25.38, 'texture_worst': 17.33, 'perimeter_worst': 184.6, 'area_worst': 2019.0,
            'smoothness_worst': 0.1622, 'compactness_worst': 0.6656, 'concavity_worst': 0.7119,
            'concave points_worst': 0.2654, 'symmetry_worst': 0.4601, 'fractal_dimension_worst': 0.1189
        }])
        prediction = predict(sample_new_data)
        print(f"Prediction for sample data: {prediction[0]} (0: Benign, 1: Malignant)")
    except FileNotFoundError as e:
        print(e)