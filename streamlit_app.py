import streamlit as st
import requests
import os

st.set_page_config(layout='wide')
st.title('Breast Cancer Prediction')

# Allow the Streamlit container to connect to the Flask API container
API_URL = os.getenv("API_URL", "http://localhost:5000/predict")

st.write('Please enter the patient\'s characteristics to get a breast cancer prediction.')

feature_info = {
    'radius_mean': {'label': 'Mean Radius', 'description': 'Average of distances from center to points on the perimeter.'},
    'texture_mean': {'label': 'Mean Texture', 'description': 'Standard deviation of gray-scale values.'},
    'perimeter_mean': {'label': 'Mean Perimeter', 'description': 'Mean size of the core tumor.'},
    'area_mean': {'label': 'Mean Area', 'description': 'Mean area of the tumor.'},
    'smoothness_mean': {'label': 'Mean Smoothness', 'description': 'Average of local variation in radius lengths.'},
    'compactness_mean': {'label': 'Mean Compactness', 'description': 'Perimeter^2 / Area - 1.0.'},
    'concavity_mean': {'label': 'Mean Concavity', 'description': 'Severity of concave portions of the contour.'},
    'concave points_mean': {'label': 'Mean Concave Points', 'description': 'Number of concave portions of the contour.'},
    'symmetry_mean': {'label': 'Mean Symmetry', 'description': 'Symmetry of the core tumor.'},
    'fractal_dimension_mean': {'label': 'Mean Fractal Dimension', 'description': '"Coastline approximation" - 1.'},
    'radius_se': {'label': 'SE Radius', 'description': 'Standard Error for the mean of distances from center to points on the perimeter.'},
    'texture_se': {'label': 'SE Texture', 'description': 'Standard Error for standard deviation of gray-scale values.'},
    'perimeter_se': {'label': 'SE Perimeter', 'description': 'Standard Error for the mean perimeter.'},
    'area_se': {'label': 'SE Area', 'description': 'Standard Error for the mean area.'},
    'smoothness_se': {'label': 'SE Smoothness', 'description': 'Standard Error for the average of local variation in radius lengths.'},
    'compactness_se': {'label': 'SE Compactness', 'description': 'Standard Error for Perimeter^2 / Area - 1.0.'},
    'concavity_se': {'label': 'SE Concavity', 'description': 'Standard Error for severity of concave portions of the contour.'},
    'concave points_se': {'label': 'SE Concave Points', 'description': 'Standard Error for number of concave portions of the contour.'},
    'symmetry_se': {'label': 'SE Symmetry', 'description': 'Standard Error for symmetry of the core tumor.'},
    'fractal_dimension_se': {'label': 'SE Fractal Dimension', 'description': 'Standard Error for "coastline approximation" - 1.'},
    'radius_worst': {'label': 'Worst Radius', 'description': '"Worst" or largest mean value for radius.'},
    'texture_worst': {'label': 'Worst Texture', 'description': '"Worst" or largest mean value for texture.'},
    'perimeter_worst': {'label': 'Worst Perimeter', 'description': '"Worst" or largest mean value for perimeter.'},
    'area_worst': {'label': 'Worst Area', 'description': '"Worst" or largest mean value for area.'},
    'smoothness_worst': {'label': 'Worst Smoothness', 'description': '"Worst" or largest mean value for smoothness.'},
    'compactness_worst': {'label': 'Worst Compactness', 'description': '"Worst" or largest mean value for compactness.'},
    'concavity_worst': {'label': 'Worst Concavity', 'description': '"Worst" or largest mean value for concavity.'},
    'concave points_worst': {'label': 'Worst Concave Points', 'description': '"Worst" or largest mean value for concave points.'},
    'symmetry_worst': {'label': 'Worst Symmetry', 'description': '"Worst" or largest mean value for symmetry.'},
    'fractal_dimension_worst': {'label': 'Worst Fractal Dimension', 'description': '"Worst" or largest mean value for fractal dimension.'}
}

input_data = {}

# Create columns for better layout
num_columns = 4

with st.form("prediction_form"):
    cols = st.columns(num_columns)

    for i, feature in enumerate(feature_info.keys()):
        with cols[i % num_columns]:
            input_data[feature] = st.number_input(feature_info[feature]['label'], value=0.0, format="%.4f", help=feature_info[feature]['description'], key=f"input_{feature}")

    submitted = st.form_submit_button('Get Prediction')
    reset_button = st.form_submit_button('Reset')

if submitted:
    try:
        response = requests.post(API_URL, json=input_data)
        if response.status_code == 200:
            result = response.json()
            st.subheader('Prediction Result:')
            
            if result['prediction'] == 1:
                st.error('**Diagnosis: Malignant**')
            else:
                st.success('**Diagnosis: Benign**')

            st.write('---') # Separator
            st.write('**Probabilities:**')
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Benign Probability", value=f"{result['probability_benign']:.4f}")
            with col2:
                st.metric(label="Malignant Probability", value=f"{result['probability_malignant']:.4f}")

        else:
            st.error(f'Error from API: {response.status_code} - {response.json().get('error', 'Unknown error')}')
    except requests.exceptions.ConnectionError:
        st.error('Could not connect to the Flask API. Please ensure it is running at http://localhost:5000.')
    except Exception as e:
        st.error(f'An unexpected error occurred: {e}')

