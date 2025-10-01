# Breast Cancer Prediction MLOps Project

This project demonstrates a foundational MLOps workflow for deploying a Breast Cancer prediction model as a Flask API. It covers project setup, model training, API development, Dockerization, and automated CI/CD with GitHub Actions.

## Project Structure

```
.venv/                  # Python virtual environment (ignored by Git)
data/                   # Stores the dataset (data.csv) - Tracked by Git for CI/CD simplicity
models/                 # Stores the trained model (model.joblib) - Ignored by Git
notebooks/              # For exploratory data analysis and experimentation
src/
  app.py              # Flask API for model inference
  model.py            # Script for training and saving the model
tests/                  # For unit and integration tests
.github/                # GitHub Actions workflow files
  workflows/
    main.yml          # CI/CD pipeline definition
Dockerfile              # Dockerfile for containerization
requirements.txt        # Python dependencies
README.md               # Project documentation
.gitignore              # Specifies files and directories to ignore in Git
streamlit_app.py        # Streamlit user interface for predictions
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

5.  **Train the model:**
    Ensure your virtual environment is activated and `data/data.csv` is present, then run:
    ```bash
    python src/model.py
    ```
    This will train the model and save it as `models/model.joblib`.

6.  **Run the Flask API locally:**
    Ensure your virtual environment is activated and the model is trained (`models/model.joblib` exists), then run:
    ```bash
    python src/app.py
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

The Streamlit application (`streamlit_app.py`) provides an interactive web interface for making predictions using the Flask API.

1.  **Run the Streamlit application locally:**
    Ensure your virtual environment is activated and the Flask API is running (as described in step 6 under "Setup and Run"), then run:
    ```bash
    streamlit run streamlit_app.py
    ```
    The UI will open in your browser, typically at `http://localhost:8501`.

## Dockerization

The project includes a `Dockerfile` to containerize the Flask API and the trained model.

1.  **Build the Docker image:**
    ```bash
    docker build -t breast-cancer-api .
    ```

2.  **Run the Docker container:**
    ```bash
    docker run -d -p 5000:5000 --name breast-cancer-api-ctr breast-cancer-api
    ```
    The API will be accessible via `http://127.0.0.1:5000/`.

3.  **Test the API (using commands from "API Usage Examples" above).**

## Automated CI/CD (GitHub Actions)

A GitHub Actions workflow (`.github/workflows/main.yml`) is configured to automate the following steps on every push to the `main` branch:

1.  **Checkout code:** Gets the latest code from the repository.
2.  **Set up Python and install dependencies:** Prepares the environment for model training.
3.  **Create `models/` directory:** Ensures the directory exists for saving the trained model.
4.  **Train the model:** Runs `src/model.py` to train the model and generate `models/model.joblib`.
5.  **Build Docker image:** Creates the Docker image for the API.
6.  **Login to Docker Hub:** Authenticates using GitHub Secrets.
7.  **Push Docker image:** Pushes the built image to your Docker Hub repository.
8.  **Run container (for testing):** Starts the newly pushed Docker image as a container.
9.  **Wait for API to be ready:** A robust loop that polls the `/` endpoint until the API responds.
10. **Test health and prediction endpoints:** Executes `curl` commands to verify API functionality.
11. **Clean up Docker container:** Stops and removes the test container.

## Future Improvements

To further enhance this MLOps project, consider these advanced steps:

1.  **Preprocessing Pipeline in `src/model.py`:** Integrate data preprocessing steps directly into a `sklearn.pipeline.Pipeline` object. Save the entire pipeline (including preprocessing and the model) to ensure consistent data transformations between training and inference.
2.  **Input Data Schema Validation in `src/app.py`:** Implement a library like `Pydantic` to define a strict and explicit schema for incoming JSON payloads to the `/predict` endpoint, providing robust data validation and clear error messages.
3.  **Multi-stage Docker Builds:** Optimize the `Dockerfile` by using multi-stage builds. The idea is to reduce the final image size by separating build-time dependencies (e.g., for training) from runtime dependencies (e.g., for serving the API).
4.  **Unit Testing:** Develop comprehensive unit tests using `pytest` for components in `src/model.py` (e.g., data loading, specific preprocessing functions) and `src/app.py` (e.g., API route logic using `Flask.test_client()`).
