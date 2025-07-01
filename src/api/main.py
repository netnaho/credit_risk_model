import os
import pandas as pd
import mlflow
from fastapi import FastAPI
from .pydantic_models import PredictionInput, PredictionOutput

# --- NEW: Direct Path Loading Method ---

# You must replace these with the NEW IDs from your 'mlruns' folder
EXPERIMENT_ID = "468728191329987875"
RUN_ID = "20362b07864e439f84330f8a3be33558"

# This constructs the exact path to the model inside the Docker container
MODEL_PATH = f"/app/mlruns/{EXPERIMENT_ID}/{RUN_ID}/artifacts/model"

print(f"Attempting to load model directly from: {MODEL_PATH}")

# Load the model directly from its folder path, bypassing the metadata bug
model = mlflow.sklearn.load_model(MODEL_PATH)

print("Model loaded successfully!")
# --- END NEW METHOD ---

# --- 3. INITIALIZE FASTAPI APP ---
app = FastAPI(
    title="Credit Risk Prediction API",
    description="API to predict credit risk probability for a customer.",
    version="1.0.0"
)

# --- 4. DEFINE THE PREDICTION ENDPOINT ---
@app.post("/predict", response_model=PredictionOutput)
def predict(data: PredictionInput):
    """
    Accepts customer features and returns the risk probability.
    """
    # Convert the Pydantic input model to a pandas DataFrame
    # The model pipeline expects a DataFrame with specific column names.
    input_df = pd.DataFrame([data.model_dump()])
    
    # Predict the probability of the positive class (is_high_risk = 1)
    risk_probability = model.predict_proba(input_df)[:, 1][0]
    
    return {"risk_probability": risk_probability}

@app.get("/")
def read_root():
    return {"status": "API is running."}