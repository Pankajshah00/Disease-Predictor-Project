import pandas as pd
import numpy as np
import pickle
import os
from sklearn.tree import DecisionTreeClassifier # A common choice for this type of problem

# Define the base directory for datasets and models
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATASETS_DIR = os.path.join(BASE_DIR, 'datasets')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Create the models directory if it doesn't exist
os.makedirs(MODELS_DIR, exist_ok=True)

print("Starting model training script...")
print(f"DEBUG: BASE_DIR is: {BASE_DIR}")
print(f"DEBUG: DATASETS_DIR is: {DATASETS_DIR}")
print(f"DEBUG: MODELS_DIR is: {MODELS_DIR}")

try:
    # Load the training data
    training_data_path = os.path.join(DATASETS_DIR, 'Training.csv')
    print(f"Attempting to load training data from: {training_data_path}")
    training_data = pd.read_csv(training_data_path)
    print("Training data loaded successfully.")

    # Separate features (X) and target (y)
    # 'prognosis' is assumed to be the target column based on your main.py
    X = training_data.drop('prognosis', axis=1)
    y = training_data['prognosis']

    # Initialize and train the Decision Tree Classifier model
    print("Training the Decision Tree Classifier model...")
    clf = DecisionTreeClassifier(random_state=42) # Using a fixed random state for reproducibility
    clf.fit(X, y)
    print("Model training complete.")

    # Define the path to save the model
    model_save_path = os.path.join(MODELS_DIR, 'disease_prediction_model.pkl')
    print(f"Attempting to save the model to: {model_save_path}")

    # Save the trained model using pickle
    with open(model_save_path, 'wb') as model_file:
        pickle.dump(clf, model_file)
    print(f"Model successfully saved as {os.path.basename(model_save_path)}")
    print("You can now try running your main.py application.")

except FileNotFoundError:
    print(f"Error: 'Training.csv' not found at {training_data_path}. Please ensure the file is in your 'datasets' directory.")
except Exception as e:
    print(f"An unexpected error occurred during model training: {e}")