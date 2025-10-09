# Breast Cancer Prediction MLOps Project

This project demonstrates a foundational MLOps workflow for deploying a Breast Cancer prediction model as a Flask API. It covers project setup, model training, API development, Dockerization, and automated CI/CD with GitHub Actions.

## Project Structure

```
breast-cancer-ops/
├── config/                    # Configuration files
│   ├── docker-compose.yml    # Defines and links multi-container Docker application
│   ├── Dockerfile.api        # Dockerfile for the Flask API container
│   └── Dockerfile.streamlit  # Dockerfile for the Streamlit UI container
├── data/                      # Stores the dataset
│   └── data.csv
├── models/                    # Stores the trained model pipeline (created locally)
│   └── model.joblib
├── src/                       # Source code
│   ├── app.py                # Flask API for model inference
│   ├── model/                # Machine Learning model components
│   │   ├── __init__.py           # Makes 'model' a Python package
│   │   ├── data_ingestion.py     # Handles raw data loading
│   │   ├── data_preprocessing.py # Contains data cleaning and feature preparation
│   │   ├── model_inference.py    # Loads trained pipeline and makes predictions
│   │   ├── model_training.py     # Orchestrates model training and pipeline saving
│   │   └── pipeline_utils.py     # Defines the scikit-learn pipeline structure
│   └── streamlit_app.py      # Streamlit user interface for predictions
├── tests/                     # For unit and integration tests
│   ├── bash_test.sh
│   ├── powershell_test.ps1
│   └── sample_payload.json
├── README.md                  # Project documentation
└── requirements.txt           # Python dependencies
```

## Setup and Run

1.  **Clone the repository (if not already done):**
    ```bash
    git clone https://github.com/Anibalrojo/breast-cancer-mlops-workflow
    cd breast-cancer-ops
    ```

2.  **Create and activate a virtual environment:**
    
    *For Windows*:
    ```powershell
    python -m venv .venv
    .\\.venv\\Scripts\\Activate.ps1
    ```

    *For Linux/macOS or Git Bash:*
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    Ensure your virtual environment is activated, then run:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Dataset Availability:**
    The `data/data.csv` dataset is committed directly to this repository. No separate download or manual directory creation is required after cloning.

5.  **Train the Machine Learning Pipeline:**
    This step will load `data/data.csv`, preprocess it, train the Random Forest classifier within a `scikit-learn` pipeline, evaluate it, and then save the complete trained pipeline to `models/model.joblib`.
    Ensure your virtual environment is activated and `data/data.csv` is present, then run:
    ```powershell
    python -m src.model.model_training
    ```
    This will train the pipeline and save it as `models/model.joblib`.

6.  **Run the Flask API locally:**
    Ensure your virtual environment is activated and the model pipeline is trained (`models/model.joblib` exists), then run:
    ```bash
    python -m src.app
    ```
    The API will be accessible at `http://127.0.0.1:5000/`. Keep this running in one terminal.

## API Usage Examples

With the Flask API running locally (as described in step 6 above), you can test its endpoints:

### 1. Health Check (`GET /`)

*   **Endpoint:** `GET http://127.0.0.1:5000/`
*   **Purpose:** Verifies that the service is running and the model is loaded.

    **PowerShell:**
    ```powershell
    Invoke-RestMethod -Uri "http://127.0.0.1:5000/" -Method Get
    ```
    **Expected Output:**
    ```json
    {
      "model_loaded": true,
      "status": "healthy"
    }
    ```

### 2. Prediction (`POST /predict`)

*   **Endpoint:** `POST http://127.0.0.1:5000/predict`
*   **Purpose:** Receives a JSON payload of features and returns a prediction.
*   **Request Body Example:** `tests/sample_payload.json`

*   **Example Usage:**

    **PowerShell:**
    ```powershell
    $headers = @{"Content-Type"="application/json"}
    $body = Get-Content -Raw -Path "tests/sample_payload.json" | ConvertFrom-Json
    Invoke-RestMethod -Uri "http://127.0.0.1:5000/predict" -Method Post -Headers $headers -Body (ConvertTo-Json $body)
    ```

    **cURL (Linux/macOS/Git Bash):**
    ```bash
    curl -X POST -H "Content-Type: application/json" -d @tests/sample_payload.json http://127.0.0.1:5000/predict
    ```

    **Expected Output:**
    ```json
    {
      "prediction": 1,
      "probability_benign": 0.1,
      "probability_malignant": 0.9
    }
    ```
    (Note: `prediction` and `probability` values will depend on your model's output for the given input.)

## Streamlit UI

The Streamlit application (`src/streamlit_app.py`) provides an interactive web interface for making predictions using the Flask API.

1.  **Run the Streamlit application locally:**
    Ensure your virtual environment is activated and the Flask API is running (as described in step 6 under "Setup and Run"), then run:
    ```bash
    streamlit run src/streamlit_app.py
    ```
    The UI will open in your browser, typically at `http://localhost:8501`.

## Dockerization

The project now uses Docker Compose to manage both the Flask API and the Streamlit UI. All Docker configuration files are located in the `config/` directory.

1.  **Build and Run with Docker Compose:**
    Ensure the model is trained (`models/model.joblib` exists), then navigate to the project root and run:
    ```bash
    docker compose -f config/docker-compose.yml up --build -d
    ```
    This will build images for `config/Dockerfile.api` and `config/Dockerfile.streamlit`, and start both services.
    The Flask API will be accessible via `http://localhost:5000/` and the Streamlit UI via `http://localhost:8501/`.

2.  **Stop Docker Compose services:**
    ```bash
    docker compose -f config/docker-compose.yml down
    ```
    This will stop and remove all services and their networks.

## Automated CI/CD (GitHub Actions)

A GitHub Actions workflow (`.github/workflows/main.yml`) is configured to automate the following steps on every push to the `main` branch:

1.  **Checkout code:** Gets the latest code from the repository.
2.  **Set up Python and install dependencies:** Prepares the environment for model training.
3.  **Create `models/` directory:** Ensures the directory exists for saving the trained model.
4.  **Train the model:** Runs `python -m src.model.model_training` to train the model and generate `models/model.joblib`.
5.  **Build and Push Docker Compose Images:** Builds both API (`config/Dockerfile.api`) and Streamlit UI (`config/Dockerfile.streamlit`) images and pushes them to Docker Hub.
6.  **Run Docker Compose services (for testing):** Starts both the API and Streamlit UI services in isolated containers.
7.  **Wait for services to be ready:** A robust loop that polls both API (`/`) and Streamlit UI (`/`) endpoints until both are responsive.
8.  **Test health and prediction endpoints:** Executes `curl` commands to verify Flask API functionality.
9.  **Test Streamlit UI is accessible:** Executes `curl` command to verify Streamlit UI responsiveness.
10. **Clean up Docker Compose services:** Stops and removes all test containers and networks.

## Future Improvements

To further enhance this MLOps project, consider these advanced steps:

1.  **Unit Testing:** Develop comprehensive unit tests using `pytest` for components in `src/model/` (e.g., data loading, specific preprocessing functions) and `src/app.py` (e.g., API route logic using `Flask.test_client()`).
2.  **Input Data Schema Validation in `src/app.py`:** Implement a library like `Pydantic` to define a strict and explicit schema for incoming JSON payloads to the `/predict` endpoint, providing robust data validation and clear error messages.
3.  **Multi-stage Docker Builds:** Optimize the `Dockerfile` by using multi-stage builds. The idea is to reduce the final image size by separating build-time dependencies (e.g., for training) from runtime dependencies (e.g., for serving the API).
