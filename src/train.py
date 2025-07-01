# src/train.py

import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer # Import the imputer

# Import our custom transformers and target engineering function
from data_processing import AggregateFeatures
from target_engineering import create_target_variable

# --- 1. SETUP MLFLOW ---
mlflow.set_experiment("Credit_Risk_Prediction")

# --- 2. LOAD AND PROCESS DATA ---
print("Loading raw data...")
try:
    raw_df = pd.read_csv('../data/raw/transactions.csv')
except FileNotFoundError:
    print("Error: 'transactions.csv' not found. Please place it in 'data/raw/'.")
    exit()

print("Engineering features...")
agg_transformer = AggregateFeatures()
features_df = agg_transformer.transform(raw_df)

print("Engineering target variable...")
target_df = create_target_variable(raw_df)

print("Merging features and target...")
model_ready_df = pd.merge(features_df, target_df, on='CustomerId')

# --- 3. PREPARE DATA FOR MODELING ---
X = model_ready_df.drop(columns=['CustomerId', 'is_high_risk'])
y = model_ready_df['is_high_risk']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- 4. DEFINE MODELS AND TRAIN ---
models = {
    "LogisticRegression": LogisticRegression(random_state=42, max_iter=1000),
    "GradientBoosting": GradientBoostingClassifier(random_state=42)
}

for model_name, model in models.items():
    print(f"--- Training {model_name} ---")
    
    with mlflow.start_run(run_name=f"{model_name}_run"):
        
        # Create a pipeline that first imputes missing values, then scales, then fits the model
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value=0)), # FILL NaNs with 0
            ('scaler', StandardScaler()),
            ('model', model)
        ])
        
        pipeline.fit(X_train, y_train)
        
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        
        mlflow.log_param("model_type", model_name)
        
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)
        
        mlflow.sklearn.log_model(pipeline, "model")
        
print("\n--- Model training and tracking complete! ---")