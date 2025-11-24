from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import os

# --- Configuration (Paths and Constants) ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

TARGET_COLUMN = 'loan_paid_back'
TEST_SIZE = 0.2
RANDOM_STATE = 42

def load_and_verify_raw_data():
    """
    Loads raw data files and checks for existence.
    """
    print ("starting data ingestion and splitting process...")
    
    # Define expected file paths
    train_path = RAW_DATA_DIR / "train.csv"
    web_test_path = RAW_DATA_DIR / "web_test.csv"
    
    if not train_path.exists() or not web_test_path.exists():
        print(f"Error: Required raw data files are missing in {RAW_DATA_DIR}")
        print("Please ensure that 'train.csv' and 'web_test.csv' are in the raw data directory.")
        return None, None
    
    # 1. Load Data
    train_df = pd.read_csv(train_path)
    web_test_df = pd.read_csv(web_test_path)
    
    print(f"Loaded training data shape: {train_df.shape}")
    print(f"Loaded web test data shape: {web_test_df.shape}")
    
    # 2. Contract Verification
    if TARGET_COLUMN not in train_df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in training data.")
    
    return train_df, web_test_df

def split_and_save_data(train_df, web_test_df):
    """
    Splits the raw training data into local train, validation, and test sets,
    and saves all necessary files to the processed directory.
    """
    
    # 1. Drop the 'id' column from the training data, keeping it for web_test_df
    train_df.drop(columns=['id'], errors='ignore', inplace = True)
    
    # 2. Split the training data into local train and test sets
    local_train_df, local_test_df = train_test_split(
        train_df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=train_df[TARGET_COLUMN]
    )
    
    # 3. Create processed data directory if it doesn't exist
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    # 4. Save the datasets
    local_train_df.to_csv(PROCESSED_DATA_DIR / "local_train.csv", index=False)
    local_test_df.to_csv(PROCESSED_DATA_DIR / "local_test.csv", index=False)
    
    print(f"\nLocal Training Set size: {local_train_df.shape}")
    print(f"Local Test Set size: {local_test_df.shape}")
    print("✅ Initial data split and ingestion complete. Files saved to data/processed/")
    

if __name__ == "__main__":
    train, web_test = load_and_verify_raw_data()
    if train is not None and web_test is not None:
        split_and_save_data(train, web_test)