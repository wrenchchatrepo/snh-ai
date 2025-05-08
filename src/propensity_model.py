# src/propensity_model.py
# Description: Trains a RandomForestClassifier to predict likelihood to buy based on age and income.
# Author: Gemini
# Date: 2024-05-07

import os
import sys
import pandas as pd
import numpy as np
import traceback
from supabase import create_client, Client
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Adjust path to import config and snh_logger from parent directory (src)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

try:
    import config
    from src import snh_logger as snh_logging
except ImportError as e:
    print(f"Critical: Could not import 'config' or 'src.snh_logger'. Error: {e}")
    import logging # Fallback
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler(sys.stderr)); logger.setLevel(logging.INFO)
    logger.critical(f"Failed to import project modules: {e}. Using basic stderr logger.")
    sys.exit("Exiting: Essential project modules failed to import.")
else:
    logger = snh_logging.get_logger(__name__)

CLEANED_TABLE_NAME = 'cleaned_customer_data'
MODEL_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "models")

# Features and Columns needed to create Target
BASE_FEATURE_COLS = ['age', 'annual_income']
TRANSACTION_COL = 'total_transactions' # Used to derive target
TARGET_COL = 'likely_to_buy'

def get_supabase_client() -> Client | None:
    """Initializes and returns a Supabase client."""
    logger.info("Initializing Supabase client for propensity_model.py.")
    if not config.SUPABASE_URL or not config.SUPABASE_SERVICE_ROLE:
        logger.error("Supabase URL or Service Role is not configured.")
        return None
    try:
        supabase_client: Client = create_client(config.SUPABASE_URL, config.SUPABASE_SERVICE_ROLE)
        logger.info("Supabase client initialized successfully.")
        return supabase_client
    except Exception as e:
        logger.error(f"Failed to initialize Supabase client: {e}", exc_info=True)
        return None

def fetch_and_prepare_data(supabase_client: Client, table_name: str) -> pd.DataFrame | None:
    """Fetches necessary data, creates target variable, handles NaNs."""
    required_cols = BASE_FEATURE_COLS + [TRANSACTION_COL]
    select_query = ", ".join(['customer_id'] + required_cols) # Include customer_id for potential reference
    logger.info(f"Attempting to fetch columns ({select_query}) from table: {table_name}")
    try:
        response = supabase_client.table(table_name).select(select_query).execute()
        if hasattr(response, 'error') and response.error:
            logger.error(f"Error fetching data from {table_name}: {response.error.message}")
            return None
        if response.data:
            df = pd.DataFrame(response.data)
            logger.info(f"Successfully fetched {len(df)} rows from {table_name}.")

            # Ensure correct types
            df['age'] = pd.to_numeric(df['age'], errors='coerce') # Keep as float/Int64 initially
            df['annual_income'] = pd.to_numeric(df['annual_income'], errors='coerce')
            df[TRANSACTION_COL] = pd.to_numeric(df[TRANSACTION_COL], errors='coerce')
            df['customer_id'] = df['customer_id'].astype(str)

            # Handle NaNs in required columns *before* creating target or selecting features
            initial_rows = len(df)
            df.dropna(subset=required_cols, inplace=True)
            if len(df) < initial_rows:
                 logger.warning(f"Dropped {initial_rows - len(df)} rows containing NaN in required columns.")

            if df.empty:
                 logger.error(f"DataFrame is empty after dropping NaN values. Cannot proceed.")
                 return pd.DataFrame()

            # Create target variable 'likely_to_buy' based on median transactions
            median_transactions = df[TRANSACTION_COL].median()
            if pd.isna(median_transactions):
                logger.error("Cannot calculate median transactions (column might be all NaN). Aborting.")
                return None
            logger.info(f"Median total_transactions: {median_transactions:.0f}. Using this as threshold for likely_to_buy.")
            df[TARGET_COL] = (df[TRANSACTION_COL] > median_transactions).astype(int)
            logger.info(f"Created target column '{TARGET_COL}'. Value counts:\\n{df[TARGET_COL].value_counts(normalize=True).round(2).to_string()}")

            # Ensure final types are appropriate (e.g., age as float for scaling)
            df['age'] = df['age'].astype(float)
            df['annual_income'] = df['annual_income'].astype(float)


            logger.info(f"Data types after fetch & target creation:\\n{df.dtypes.to_string()}")
            return df
        else:
            logger.info(f"No data found in {table_name}.")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Unexpected error fetching/preparing data from {table_name}: {e}", exc_info=True)
        return None

def main():
    """
    Main function: Fetch data, preprocess, train, evaluate classifier, save artifacts.
    """
    logger.info("--- Starting propensity_model.py script execution ---")

    supabase = get_supabase_client()
    if supabase is None:
        logger.critical("Supabase client initialization failed. Aborting.")
        sys.exit(1)

    propensity_df = fetch_and_prepare_data(supabase, CLEANED_TABLE_NAME)
    if propensity_df is None or propensity_df.empty:
        logger.critical(f"Failed to fetch/prepare data or data is empty from {CLEANED_TABLE_NAME}. Aborting.")
        sys.exit(1)

    # Define Features (X) and Target (y)
    X = propensity_df[BASE_FEATURE_COLS]
    y = propensity_df[TARGET_COL]
    customer_ids = propensity_df['customer_id'] # Keep IDs for potential prediction saving

    logger.info(f"Features shape: {X.shape}, Target shape: {y.shape}")

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=config.RANDOM_STATE, stratify=y)
    logger.info(f"Split data: Train={len(X_train)}, Test={len(X_test)}")

    # Scale numerical features (age, annual_income)
    # Fit scaler ONLY on training data
    scaler = StandardScaler()
    logger.info("Fitting StandardScaler on training data...")
    X_train_scaled = scaler.fit_transform(X_train)
    logger.info("Transforming training and test data...")
    X_test_scaled = scaler.transform(X_test)

    # Train RandomForestClassifier
    logger.info("Training RandomForestClassifier...")
    rfc_model = RandomForestClassifier(n_estimators=100, # Default, can be tuned
                                     random_state=config.RANDOM_STATE,
                                     class_weight='balanced', # Good for potentially imbalanced target
                                     n_jobs=-1)
    try:
        rfc_model.fit(X_train_scaled, y_train)
        logger.info("Model training complete.")
    except Exception as e:
        logger.error(f"Error during model training: {e}", exc_info=True)
        sys.exit(1) # Stop if training fails

    # Evaluate model
    logger.info("Evaluating model on test set...")
    try:
        y_pred = rfc_model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=['Not Likely (0)', 'Likely (1)'])
        cm = confusion_matrix(y_test, y_pred)

        logger.info(f"Evaluation results for {TARGET_COL}:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Classification Report:\\n{report}")
        logger.info(f"  Confusion Matrix:\\n{cm}")

    except Exception as e:
        logger.error(f"Error during model evaluation: {e}", exc_info=True)

    # Save the trained model and scaler
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    model_filename = os.path.join(MODEL_OUTPUT_DIR, "propensity_model_rfc.joblib")
    scaler_filename = os.path.join(MODEL_OUTPUT_DIR, "propensity_scaler.joblib")
    try:
        joblib.dump(rfc_model, model_filename)
        joblib.dump(scaler, scaler_filename)
        logger.info(f"Saved trained classifier model to {model_filename}")
        logger.info(f"Saved fitted scaler to {scaler_filename}")
    except Exception as e:
        logger.error(f"Error saving model or scaler: {e}", exc_info=True)

    # Optional: Generate predictions for the whole dataset and save (similar to predictive_model.py)
    # This part is skipped for now as per keeping it clean.

    logger.info("--- propensity_model.py script execution finished ---")

if __name__ == "__main__":
    main()
