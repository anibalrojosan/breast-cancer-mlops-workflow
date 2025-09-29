import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

def train_model(data_path='data/data.csv', model_path='models/model.joblib'):
    # Load the dataset
    df = pd.read_csv(data_path)

    # Basic preprocessing: Drop the 'id' column and the 'Unnamed: 32' column
    # which often appears as an empty column in this dataset
    df.drop(['id', 'Unnamed: 32'], axis=1, inplace=True)

    # Convert 'diagnosis' column to numerical (M=1, B=0)
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

    # Define features (X) and target (y)
    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a RandomForestClassifier model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model (optional, but good for MLOps)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.4f}")

    # Ensure the models directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    # Save the trained model using joblib
    joblib.dump(model, model_path)
    print(f"Model trained and saved to {model_path}")

if __name__ == "__main__":
    train_model()
