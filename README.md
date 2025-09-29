# Breast Cancer Prediction MLOps Project

This project demonstrates a basic MLOps workflow for deploying a Breast Cancer prediction model as a Flask API.

## Project Structure

```
.venv/
data/                   # Stores the dataset (data.csv)
models/                 # Stores the trained model (model.joblib)
src/
  app.py                # Flask API for model inference
  model.py              # Script for training and saving the model
tests/                  # For unit and integration tests
Dockerfile              # Dockerfile for containerization
requirements.txt        # Python dependencies
README.md               # Project documentation
```

## Setup and Run

1.  **Clone the repository (if not already done):**
    ```bash
    git clone https://github.com/Anibalrojo/breast-cancer-mlops-workflowy
    cd breast-cancer-ops
    ```

2.  **Create and activate a virtual environment:**
    
    *For Windows*:
    ```powershell
    python -m venv .venv
    .venv\Scripts\Activate.ps1
    ```

    *For Linux/macOS or Git Bash:*
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download the dataset:**
    Manually download the `data.csv` file from [Kaggle: Breast Cancer Wisconsin (Diagnostic) Data Set](https://www.kaggle.com/datasets/uciml/breast-cancerwisconsin-data).
    **Create a `data/` directory** at the project root and place the downloaded `data.csv` file inside it.

5.  **Train the model:**
    ```bash
    python src/model.py
    ```
    This will train the model and save it as `models/model.joblib`.

## To be added in a later step:
1. Create the Flask API
2. Containerization using Docker
3. CI/CD Automatization
