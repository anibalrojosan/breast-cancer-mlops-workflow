# Breast Cancer Prediction MLOps Project

This project demonstrates a basic MLOps workflow for deploying a Breast Cancer prediction model as a Flask API.

## Project Structure

```
.venv/
data/                   # Stores the dataset (data.csv)
models/                 # Stores the trained model (model.joblib)
notebooks/              # For exploratory data analysis and experimentation
src/
  app.py              # Flask API for model inference
  model.py            # Script for training and saving the model
tests/                  # For unit and integration tests
Dockerfile              # Dockerfile for containerization
requirements.txt        # Python dependencies
README.md               # Project documentation
```

## Setup and Run

1.  **Clone the repository (if not already done):**
    ```bash
    git clone <your-repo-url>
    cd breast-cancer-ops
    ```

2.  **Create and activate a virtual environment:**
    ```powershell
    python -m venv .venv
    .venv\Scripts\Activate.ps1
    ```

3.  **Install dependencies:**
    ```powershell
    pip install -r requirements.txt
    ```

4.  **Download the dataset:**
    Manually download the `data.csv` file from [Kaggle: Breast Cancer Wisconsin (Diagnostic) Data Set](https://www.kaggle.com/datasets/uciml/breast-cancerwisconsin-data) and place it in the `data/` directory.

5.  **Train the model:**
    ```powershell
    python src/model.py
    ```
    This will train the model and save it as `models/model.joblib`.

6.  **Run the Flask API locally:**
    With the virtual environment activated:
    ```powershell
    python src/app.py
    ```
    The API will be accessible at `http://127.0.0.1:5000/`.

    *To test the health check:*
    Open your browser to `http://127.0.0.1:5000/`.

    *To test the prediction endpoint (using PowerShell):*
    ```powershell
    $headers = @{"Content-Type"="application/json"}
    $body = @{
        "radius_mean"=17.99;
        "texture_mean"=10.38;
        "perimeter_mean"=122.8;
        "area_mean"=1001.0;
        "smoothness_mean"=0.1184;
        "compactness_mean"=0.2776;
        "concavity_mean"=0.3001;
        "concave points_mean"=0.1471;
        "symmetry_mean"=0.2419;
        "fractal_dimension_mean"=0.07871;
        "radius_se"=1.095;
        "texture_se"=0.3568;
        "perimeter_se"=8.589;
        "area_se"=153.4;
        "smoothness_se"=0.006399;
        "compactness_se"=0.04904;
        "concavity_se"=0.05373;
        "concave points_se"=0.01587;
        "symmetry_se"=0.03003;
        "fractal_dimension_se"=0.006193;
        "radius_worst"=25.38;
        "texture_worst"=17.33;
        "perimeter_worst"=184.6;
        "area_worst"=2019.0;
        "smoothness_worst"=0.1622;
        "compactness_worst"=0.6656;
        "concavity_worst"=0.7119;
        "concave points_worst"=0.2654;
        "symmetry_worst"=0.4601;
        "fractal_dimension_worst"=0.1189
    }
    Invoke-RestMethod -Uri "http://127.0.0.1:5000/predict" -Method Post -Headers $headers -Body (ConvertTo-Json $body)
    ```

## Dockerization

(To be added in a later step)

## CI/CD Automatization

(To be added in a later step)
