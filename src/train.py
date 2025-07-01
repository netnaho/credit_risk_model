# src/train.py

# --- SELF-VERIFICATION STEP ---
print("--- Verifying Source Code of train.py ---")
try:
    with open(__file__, 'r') as f:
        print(f.read())
except Exception as e:
    print(f"Could not read source file: {e}")
print("--- Verification Complete ---\n")
# --- END VERIFICATION ---


import pandas as pd
import mlflow
import mlflow.sklearn
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Import custom transformers
from data_processing import AggregateFeatures
from target_engineering import create_target_variable

# Define paths robustly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / 'data' / 'raw' / 'transactions.csv'

# Set experiment name. MLflow will create 'mlruns' in the current directory (project root).
mlflow.set_experiment("Credit_Risk_Prediction")

# Load Data
print("Loading raw data...")
raw_df = pd.read_csv(DATA_PATH)

# Feature & Target Engineering
print("Engineering features and target...")
features_df = AggregateFeatures().transform(raw_df)
target_df = create_target_variable(raw_df)
model_ready_df = pd.merge(features_df, target_df, on='CustomerId')

# Prepare Data for Modeling
X = model_ready_df.drop(columns=['CustomerId', 'is_high_risk'])
y = model_ready_df['is_high_risk']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define and Train Models
models = {
    "LogisticRegression": LogisticRegression(random_state=42, max_iter=1000),
    "GradientBoosting": GradientBoostingClassifier(random_state=42)
}

for model_name, model in models.items():
    print(f"--- Training {model_name} ---")
    with mlflow.start_run(run_name=f"{model_name}_run"):
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
            ('scaler', StandardScaler()),
            ('model', model)
        ])
        pipeline.fit(X_train, y_train)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
        print(f"ROC AUC: {roc_auc:.4f}")

        mlflow.log_param("model_type", model_name)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.sklearn.log_model(pipeline, "model")

print("\n--- Model training and tracking complete! ---")