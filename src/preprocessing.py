import pandas as pd
from pathlib import Path
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, FunctionTransformer
import numpy as np

# --- 1. CONFIGURATION AND CONSTANTS ---

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
PIPELINE_DIR = PROJECT_ROOT / "models" / "pipelines"
PIPELINE_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COLUMN = 'loan_paid_back'

# --- Feature Lists ---

# 1. Features requiring Log Transform + Scaling
LOG_NUMERICAL_FEATURES = ['annual_income']

# 2. Features requiring ONLY Scaling
BASIC_NUMERICAL_FEATURES = [
    'debt_to_income_ratio', 
    'credit_score', 
    'loan_amount', 
    'interest_rate'
]

# 3. Nominal features requiring One-Hot Encoding
NOMINAL_FEATURES = [
    'gender', 
    'marital_status', 
    'education_level',
    'employment_status', 
    'loan_purpose'
]

# 4. Ordinal features requiring Custom Mapping
ORDINAL_FEATURES = [
    'grade_subgrade'
]

# --- Custom Mappers ---
GRADE_SUBGRADE_ORDER = [f'{g}{s}' for g in 'ABCDEFG' for s in '12345']
ORDINAL_MAPPERS = {
    'grade_subgrade': GRADE_SUBGRADE_ORDER
}


# --- 2. TRANSFORMATION LOGIC ---

def log_transform(X):
    return np.log1p(X)

# FIX: Added feature_names_out="one-to-one" to suppress warning and handle column names
LOG_TRANSFORMER = FunctionTransformer(
    log_transform, 
    validate=True, 
    feature_names_out="one-to-one"
)


# --- 3. MODULAR BUILDERS ---

def get_numerical_transformers(strategy: str) -> list:
    """Returns the list of numerical transformer tuples based on strategy."""
    transformers = []
    
    # Define the pipelines explicitly here
    log_pipeline = Pipeline(steps=[
        ('log', LOG_TRANSFORMER),
        ('scaler', StandardScaler())
    ])
    
    standard_pipeline = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    # Assign them based on strategy
    if strategy == 'targeted':
        # Use log pipe for income, standard pipe for the rest
        transformers.append(('num_log', log_pipeline, LOG_NUMERICAL_FEATURES))
        transformers.append(('num_basic', standard_pipeline, BASIC_NUMERICAL_FEATURES))
        
    elif strategy == 'baseline':
        # Use standard pipe for EVERYTHING
        all_numericals = LOG_NUMERICAL_FEATURES + BASIC_NUMERICAL_FEATURES
        transformers.append(('num_all', standard_pipeline, all_numericals))
        
    return transformers


def get_categorical_transformers(strategy: str) -> list:
    """Returns the list of categorical transformer tuples based on strategy."""
    transformers = []
    
    # Define encoders explicitly here
    ordinal_encoder = OrdinalEncoder(
        categories=[ORDINAL_MAPPERS['grade_subgrade']], 
        handle_unknown='use_encoded_value', 
        unknown_value=-1
    )
    
    onehot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    # Assign them based on strategy
    if strategy == 'targeted':
        # Use Ordinal for Grade, OneHot for the rest
        transformers.append(('ordinal', ordinal_encoder, ORDINAL_FEATURES))
        transformers.append(('onehot', onehot_encoder, NOMINAL_FEATURES))
        
    elif strategy == 'baseline':
        # Use OneHot for EVERYTHING (Ignore Ranks)
        all_categoricals = ORDINAL_FEATURES + NOMINAL_FEATURES
        transformers.append(('onehot_all', onehot_encoder, all_categoricals))
        
    return transformers


# --- 4. PIPELINE FACTORY ---

def get_feature_pipelines(strategy: str = 'baseline') -> Pipeline:
    """
    Assembles the final pipeline by combining modular transformer lists.
    """
    # Load Data (needed to initialize the ColumnTransformer structure)
    train_df = pd.read_csv(PROCESSED_DATA_DIR / "local_train.csv")
    X = train_df.drop(columns=[TARGET_COLUMN])
    y = train_df[TARGET_COLUMN]

    # Combine the lists from our builders
    # This keeps numerical and categorical logic separate but joins them at the end
    all_transformers = get_numerical_transformers(strategy) + get_categorical_transformers(strategy)
    
    # Build the ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=all_transformers,
        remainder='drop',
        verbose_feature_names_out=False
    )
    
    preprocessor.set_output(transform="pandas")

    # Wrap in Pipeline
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor)
    ])
    
    return full_pipeline, X, y


# --- 5. EXECUTION ---

def save_pipeline(pipeline: Pipeline, file_name: str):
    path = PIPELINE_DIR / file_name
    joblib.dump(pipeline, path)
    print(f"✅ Pipeline saved to {path}")

if __name__ == "__main__":
    # 1. Baseline
    print("\n--- Building Baseline Pipeline ---")
    pipe_base, X, y = get_feature_pipelines(strategy='baseline')
    pipe_base.fit(X, y)
    save_pipeline(pipe_base, "baseline_pipeline.joblib")
    
    # 2. Targeted
    print("\n--- Building Targeted Pipeline ---")
    pipe_target, X, y = get_feature_pipelines(strategy='targeted')
    pipe_target.fit(X, y)
    save_pipeline(pipe_target, "targeted_pipeline.joblib")
    
    print("\nSuccess! Feature pipelines are ready for Phase 3.")