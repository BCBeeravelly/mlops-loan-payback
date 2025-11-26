import pandas as pd
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, FunctionTransformer
import numpy as np

# --- Configuration (Paths and Constants) ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
PIPELINE_DIR = PROJECT_ROOT / "models" / "pipelines"
PIPELINE_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COLUMN = 'loan_paid_back'
RANDOM_STATE = 42

# --- Feature Lists ---
NUMERICAL_FEATURES = ['annual_income', 'debt_to_income_ratio', 'credit_score', 'loan_amount', 'interest_rate']
NOMINAL_FEATURES = ['gender', 'marital_status', 'education_level','employment_status', 'loan_purpose']
ORDINAL_FEATURES = ['grade_subgrade']

GRADE_SUBGRADE_ORDER = [f'{g}{s}' for g in 'ABCDEFG' for s in '12345']
ORDINAL_MAPPERS = {
    'grade_subgrade': GRADE_SUBGRADE_ORDER
}

# Custom log transform
def log_transform(X):
    return np.log1p(X)

LOG_TRANSFORMER = FunctionTransformer(log_transform, validate=True)

def get_numerical_pipeline(use_log_transform: bool = False) -> Pipeline:
    """Returns a pipeline for numerical features, optionally including log transformation."""
    steps = []
    if use_log_transform:
        steps.append(('log_transform', LOG_TRANSFORMER))
    steps.append(('scaler', StandardScaler()))
    return Pipeline(steps=steps)

def get_categorical_pipeline(use_ordinal:bool = False) -> ColumnTransformer:
    """
    Returns the full feature engineering pipeline (ColumnTransformer).
    
    If use_ordinal=True, OrdinalEncoder is used for 'grade_subgrade'.
    Otherwise, OneHotEncoder is used for all categorical features.
    """
    
    transformers = []
    
    # 1. Numerical Pipeline (Always Scaled)
    numerical_pipe = get_numerical_pipeline()
    
    if use_ordinal:
        # Targeted Strategy
        ordinal_encoder = OrdinalEncoder(categories=[
            ORDINAL_MAPPERS['grade_subgrade']
        ], handle_unknown='use_encoded_value', unknown_value=-1)
        
        transformers.append(('ordinal', ordinal_encoder, ORDINAL_FEATURES))
        transformers.append(('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False), NOMINAL_FEATURES))
        
    else:
        # Baseline Strategy
        all_nominal_and_ordinal = NOMINAL_FEATURES + ORDINAL_FEATURES
        transformers.append(('onehot_all', OneHotEncoder(handle_unknown='ignore', sparse_output=False), all_nominal_and_ordinal))
        
    # Add numerical preprocessing
    numerical_features_to_scale = NUMERICAL_FEATURES
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipe, numerical_features_to_scale),
        ] + transformers,
        remainder='drop',
        verbose_feature_names_out=False
    )
    
    preprocessor.set_output(transform='pandas')
    return preprocessor

# --- Pipeline Factory ---

def get_feature_pipelines(strategy: str = 'baseline') -> Pipeline:
    """
    The main factory function that returns a full ML Pipeline object based on strategy.
    
    Args:
        strategy (str): 'baseline', 'targeted' or 'derived_1'
    """
    
    train_df = pd.read_csv(PROCESSED_DATA_DIR / "local_train.csv")
    X = train_df.drop(columns=[TARGET_COLUMN])
    y = train_df[TARGET_COLUMN]
    
    # --- Preprocessor Selection ---
    if strategy == 'targeted':
        # Targeted Pipeline: Log Transform Income + Ordinal for Grade
        numerical_pipe_with_log = get_numerical_pipeline(use_log_transform=True)
        preprocessor = get_categorical_pipeline(use_ordinal=True)
        
        # Override the numerical transformer in the ColumnTransformer
        preprocessor.transformers[0] = ('num', numerical_pipe_with_log, NUMERICAL_FEATURES)
        
    elif strategy == 'baseline':
        # Baseline Pipeline: No log transform, all categoricals as nominal
        preprocessor = get_categorical_pipeline(use_ordinal=False)
    
    elif strategy == 'derived_1':
        
        numerical_pipe_with_log = get_numerical_pipeline(use_log_transform=True)
        preprocessor = get_categorical_pipeline(use_ordinal=True)
        preprocessor.transformers[0] = ('num', numerical_pipe_with_log, NUMERICAL_FEATURES)
        
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # --- Create the Full Pipeline (Preprocessor + Estimator Placeholder) ---
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor)
    ])
    
    return full_pipeline, X, y

# --- Execution and Saving ---
def save_pipeline(pipeline: Pipeline, file_name: str):
    """Saves the fitted pipeline object using joblib."""
    path = PIPELINE_DIR / file_name
    joblib.dump(pipeline, path)
    print(f"Pipeline saved to {path}")
    
if __name__ == "__main__":
    
    PIPELINE_DIR.mkdir(parents=True, exist_ok=True)
    
    # --- Execute and Test the Baseline Pipeline ---
    print("\n--- Testing Baseline Pipeline Creation ---")
    pipeline_baseline, X_train, y_train = get_feature_pipelines(strategy='baseline')
    pipeline_baseline.fit(X_train, y_train)
    save_pipeline(pipeline_baseline, "baseline_pipeline.joblib")
    print("✅ Feature engineering pipelines created and saved successfully.")
    
     # --- Execute and Test the Targeted Pipeline ---
    
    print("\n--- Testing Targeted Pipeline Creation ---")
    pipeline_targeted, X_train, y_train = get_feature_pipelines(strategy='targeted')
    pipeline_targeted.fit(X_train, y_train)
    save_pipeline(pipeline_targeted, "targeted_pipeline.joblib")
    print("✅ Feature engineering pipelines created and saved successfully.")
    
    print("\nFeature processing setup complete. Two pipelines saved in models/pipelines/")