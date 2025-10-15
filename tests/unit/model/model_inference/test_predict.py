import pandas as pd
import pytest
import src.model.model_inference as model_inference

def test_predict_function_with_mocker(mocker):
    sample_data = pd.DataFrame([{
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
    
    expected_prediction = [0]

    # Create a mock for the pipeline object that the 'load_pipeline' function in 
    # model_inference.py would return
    mock_pipeline = mocker.MagicMock()
    mock_pipeline.predict.return_value = expected_prediction

    # Patch 'load_pipeline' to return our mock pipeline
    mock_load_pipeline = mocker.patch('src.model.model_inference.load_pipeline', return_value=mock_pipeline)

    # Call the predict function using the module reference
    prediction_result = model_inference.predict(sample_data, model_path="dummy/path.joblib")

    # Assertions
    # 1. The 'load_pipeline' function was called once with the correct path
    mock_load_pipeline.assert_called_once_with("dummy/path.joblib")
    # 2. The 'predict' function of the mock pipeline was called once with the correct data
    mock_pipeline.predict.assert_called_once_with(sample_data)
    # 3. The prediction result is as expected
    assert prediction_result == expected_prediction