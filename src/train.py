import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

# Import pipeline factor
from src.preprocessing import get_feature_pipelines, TARGET_COLUMN

import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "local_train.csv"
EXPERIMENT_NAME = "Loan_Default_Prediction"

def train_model(strategy="baseline", penalty="l2", C=1.0, class_weight=None):
    """
    Trains a Loistic Regression model with the specified feature strategy and hyperparameters
    """
    
    # 1. Setup MLflow
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    with mlflow.start_run():
        print(f'Starting Run: Strategy={strategy}, C={C}')
        
        # 2. Log Hyperparameters
        params = {
            "strategy": strategy,
            "model": "LogisticRegression",
            "penalty": penalty,
            "C": C,
            "class_weight": class_weight
        }
        mlflow.log_params(params)
        
        # 3. Load Data & Build Pipeline
        # The factory returns the pipeline structure and the X, y data
        pipeline, X, y = get_feature_pipelines(strategy=strategy)
        
        # 4. Split Data (Train/Validation)
        # Split inside the run to ensure we validate on unseen data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # 5. Add Model to Pipeline
        model = LogisticRegression(
            penalty=penalty,
            C=C,
            class_weight=class_weight,
            solver='lbfgs',
            max_iter=1000,
            random_state=42
        )
        pipeline.steps.append(('classifier', model))
        
        # 6. Train
        print("Training model...")
        pipeline.fit(X_train, y_train)
        
        # 7. Evaluate
        print("Evaluating...")
        y_pred = pipeline.predict(X_val)
        y_prob = pipeline.predict_proba(X_val)[:, 1]
        
        metrics = {
            "accuracy": accuracy_score(y_val, y_pred),
            "precision": precision_score(y_val, y_pred, zero_division=0),
            "recall": recall_score(y_val, y_pred),
            "roc_auc": roc_auc_score(y_val, y_prob)
        }
        
        # 8. Log metrics & model
        mlflow.log_metrics(metrics)
        signature = infer_signature(X_val, y_pred)
        
        ## Create an input example
        input_example = X_val.iloc[:5]
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            signature=signature,
            input_example=input_example)
        
        print(f"Run complete. Accuracy: {metrics['accuracy']:.4f}, AUC: {metrics['roc_auc']:.4f}")
        print("-"*30)
        
if __name__ == "__main__":
    # Experiment 1: Baseline Strategy (No Log Transform)
    train_model(strategy="baseline", C=1.0)
    
    # Experiment 2: Targeted Strategy (Log + Ordinal)
    train_model(strategy="targeted", C=1.0)
    
    # Experiment 3: Targeted + Balanced Class Weights (Handling Imbalance)
    train_model(strategy="targeted", C=1.0, class_weight="balanced")