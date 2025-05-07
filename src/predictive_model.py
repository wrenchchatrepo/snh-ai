```python
# src/predictive_model.py
# Description: Trains a RandomForestRegressor model to predict total transactions.
# Author: Gemini
# Date: 2024-05-07

import os
import sys
import pandas as pd
import numpy as np
from supabase import create_client, Client
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib # For saving models optionally

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
MODEL_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "models") # Optional: Directory to save trained models

# Features and Targets
FEATURE_COLS = ['age', 'region']
# TARGET_COL_INCOME = 'annual_income' # Removed income prediction
TARGET_COL_TRANSACTIONS = 'total_transactions'

def get_supabase_client() -> Client | None:
    """Initializes and returns a Supabase client."""
    logger.info("Initializing Supabase client for predictive_model.py.")
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

def fetch_cleaned_data(supabase_client: Client, table_name: str) -> pd.DataFrame | None:
    """Fetches necessary data from the cleaned data table."""
    required_cols = FEATURE_COLS + [TARGET_COL_TRANSACTIONS] # Removed TARGET_COL_INCOME
    select_query = ", ".join(required_cols)
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
            df['age'] = pd.to_numeric(df['age'], errors='coerce').astype('Int64')
            df['region'] = df['region'].astype(str)
            # df[TARGET_COL_INCOME] = pd.to_numeric(df[TARGET_COL_INCOME], errors='coerce') # Removed income target processing
            df[TARGET_COL_TRANSACTIONS] = pd.to_numeric(df[TARGET_COL_TRANSACTIONS], errors='coerce').astype('Int64')

            # Handle potential NaNs introduced by coercion or missing values from cleaning
            initial_rows = len(df)
            df.dropna(subset=required_cols, inplace=True)
            if len(df) < initial_rows:
                 logger.warning(f"Dropped {initial_rows - len(df)} rows containing NaN in required columns.")

            logger.info(f"Data types after fetch & cleaning NaNs:\\n{df.dtypes.to_string()}")
            return df
        else:
            logger.info(f"No data found in {table_name}.")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Unexpected error fetching data from {table_name}: {e}", exc_info=True)
        return None

def train_evaluate_model(df: pd.DataFrame, feature_cols: list, target_col: str, model_name: str):
    """Prepares data, trains a RandomForestRegressor, and evaluates it."""
    logger.info(f"--- Training and evaluating model for target: {target_col} ---")

    if df.empty:
        logger.error(f"Input data for {target_col} model is empty. Skipping.")
        return None

    X = df[feature_cols]
    y = df[target_col]

    # Define preprocessing steps (OneHotEncode region, passthrough age)
    # Note: Scaling 'age' might be beneficial for some models, but RF is less sensitive.
    # Keep it simple for now.
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=int), ['region'])
        ],
        remainder='passthrough' # Keeps 'age'
    )

    # Split data (optional but good practice)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=config.RANDOM_STATE)
    logger.info(f"Split data for {target_col}: Train={len(X_train)}, Test={len(X_test)}")

    # Fit preprocessor and transform data
    logger.info("Fitting preprocessor and transforming data...")
    try:
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        # Get feature names after OHE for clarity if needed
        ohe_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(['region'])
        # Determine passthrough columns order might be tricky, assume 'age' comes after OHE columns
        passthrough_features = [col for col in feature_cols if col not in ['region']] # Assuming 'age' is the only one
        processed_feature_names = np.concatenate([ohe_feature_names, passthrough_features])
        logger.info(f"Processed feature names: {processed_feature_names.tolist()}")
        logger.info(f"Shape of processed training features: {X_train_processed.shape}")
    except Exception as e:
         logger.error(f"Error during preprocessing: {e}", exc_info=True)
         return None

    # Train RandomForestRegressor model
    logger.info("Training RandomForestRegressor...")
    rf_model = RandomForestRegressor(n_estimators=100, # Default, can be tuned
                                   random_state=config.RANDOM_STATE,
                                   n_jobs=-1) # Use all available CPU cores
    try:
        rf_model.fit(X_train_processed, y_train)
        logger.info("Model training complete.")
    except Exception as e:
        logger.error(f"Error during model training: {e}", exc_info=True)
        return None

    # Evaluate model
    logger.info("Evaluating model on test set...")
    try:
        y_pred = rf_model.predict(X_test_processed)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        logger.info(f"Evaluation results for {target_col}:")
        logger.info(f"  Mean Squared Error (MSE): {mse:.2f}")
        logger.info(f"  R-squared (R2 ): {r2:.4f}")
    except Exception as e:
        logger.error(f"Error during model evaluation: {e}", exc_info=True)
        # Continue even if evaluation fails, model is still trained

    # Optional: Save the trained model and preprocessor
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    model_filename = os.path.join(MODEL_OUTPUT_DIR, f"{model_name}_model.joblib")
    preprocessor_filename = os.path.join(MODEL_OUTPUT_DIR, f"{model_name}_preprocessor.joblib")
    try:
        joblib.dump(rf_model, model_filename)
        joblib.dump(preprocessor, preprocessor_filename)
        logger.info(f"Saved trained model to {model_filename}")
        logger.info(f"Saved preprocessor to {preprocessor_filename}")
    except Exception as e:
        logger.error(f"Error saving model or preprocessor: {e}", exc_info=True)


    logger.info(f"--- Finished processing model for target: {target_col} ---")
    return rf_model, preprocessor # Return trained objects if needed

def main():
    """
    Main function: Fetch data, train and evaluate prediction models.
    """
    logger.info("--- Starting predictive_model.py script execution ---")

    supabase = get_supabase_client()
    if supabase is None:
        logger.critical("Supabase client initialization failed. Aborting.")
        sys.exit(1)

    cleaned_df = fetch_cleaned_data(supabase, CLEANED_TABLE_NAME)
    if cleaned_df is None or cleaned_df.empty:
        logger.critical(f"Failed to fetch or data is empty from {CLEANED_TABLE_NAME}. Aborting.")
        sys.exit(1)

    # Train model for total_transactions
    train_evaluate_model(cleaned_df, FEATURE_COLS, TARGET_COL_TRANSACTIONS, "transactions_predictor")

    logger.info("--- predictive_model.py script execution finished ---")

if __name__ == "__main__":
    main()
```