```python
# src/predictive_model.py
# Description: Trains a RandomForestRegressor model to predict total transactions and saves predictions.
# Date: 2024-05-07

import os
import sys
import pandas as pd
import numpy as np
import traceback
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
PREDICTIONS_TABLE_NAME = 'transaction_predictions'
MODEL_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "models")
REPLACE_EXISTING_PREDICTIONS = True # Flag to control replacement

# Features and Targets
FEATURE_COLS = ['age', 'region']
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
    # Ensure customer_id is always fetched along with features and target
    required_cols = ['customer_id'] + FEATURE_COLS + [TARGET_COL_TRANSACTIONS]
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
            df['customer_id'] = df['customer_id'].astype(str) # Ensure PK is string
            df['age'] = pd.to_numeric(df['age'], errors='coerce').astype('Int64')
            df['region'] = df['region'].astype(str)
            df[TARGET_COL_TRANSACTIONS] = pd.to_numeric(df[TARGET_COL_TRANSACTIONS], errors='coerce').astype('Int64')

            # Handle potential NaNs introduced by coercion or missing values from cleaning
            initial_rows = len(df)
            df.dropna(subset=required_cols, inplace=True)
            if len(df) < initial_rows:
                 logger.warning(f"Dropped {initial_rows - len(df)} rows containing NaN in required columns.")

            if df.empty:
                 logger.error(f"DataFrame is empty after dropping NaN values. Cannot proceed.")
                 return pd.DataFrame()

            logger.info(f"Data types after fetch & cleaning NaNs:\\n{df.dtypes.to_string()}")
            return df
        else:
            logger.info(f"No data found in {table_name}.")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"Unexpected error fetching data from {table_name}: {e}", exc_info=True)
        return None

# --- START: Added Helper Functions ---
def delete_all_data_from_table(table_name: str, supabase_client: Client) -> bool:
    """Deletes all data from the specified Supabase table."""
    logger.info(f"Attempting to delete all data from Supabase table: {table_name}.")
    try:
        # Use a filter that matches all rows, adjusting column if needed
        response = supabase_client.table(table_name).delete().neq('customer_id', '__non_existent_value_for_delete_all__').execute()
        if hasattr(response, 'error') and response.error:
            # Check for specific error like 'relation "..." does not exist'
            if 'does not exist' in response.error.message:
                 logger.warning(f"Table '{table_name}' does not seem to exist. Skipping delete.")
                 return True # Treat as success if table doesn't exist
            logger.error(f"Error deleting data from Supabase table '{table_name}': {response.error.message}")
            return False
        deleted_count = len(response.data) if hasattr(response, 'data') and response.data else "an unknown number of"
        logger.info(f"Successfully deleted {deleted_count} records from '{table_name}' (or table was empty/non-existent).")
        return True
    except Exception as e:
        logger.error(f"An unexpected error occurred during data deletion from table '{table_name}': {e}", exc_info=True)
        return False

def load_data_to_supabase(df: pd.DataFrame, table_name: str, supabase_client: Client) -> bool:
    """Loads prediction data from a pandas DataFrame into the specified Supabase table."""
    if df.empty:
        logger.info(f"Predictions DataFrame is empty. No data to load into Supabase table '{table_name}'.")
        return True

    logger.info(f"Attempting to load {len(df)} rows into Supabase table: {table_name}.")

    # Ensure correct types before converting to dicts
    if 'predicted_total_transactions' in df.columns:
         # Convert to standard float (handles potential NaN if needed)
         df['predicted_total_transactions'] = pd.to_numeric(df['predicted_total_transactions'], errors='coerce')
    if 'customer_id' in df.columns:
         df['customer_id'] = df['customer_id'].astype(str)

    records = df.astype(object).where(pd.notnull(df), None).to_dict(orient='records')
    try:
        response = supabase_client.table(table_name).insert(records).execute()
        if hasattr(response, 'error') and response.error:
            logger.error(f"Error inserting data into Supabase table '{table_name}': {response.error.message}")
            if hasattr(response.error, 'details'): logger.error(f"Details: {response.error.details}")
            if hasattr(response.error, 'hint'): logger.error(f"Hint: {response.error.hint}")
            return False
        elif hasattr(response, 'data') and response.data: # Check if data is returned on success
            logger.info(f"Successfully inserted {len(response.data)} records into '{table_name}'.")
            return True
        elif not (hasattr(response, 'error') and response.error): # No error reported
             logger.info(f"Supabase insert for table '{table_name}' completed successfully (no error reported).")
             return True
        else: # Unexpected response
            logger.warning(f"Supabase insert for table '{table_name}' resulted in an unexpected response format.")
            logger.debug(f"Full Supabase response: {response}")
            return False
    except Exception as e:
        logger.error(f"An unexpected error occurred during Supabase data loading for '{table_name}': {e}", exc_info=True)
        return False
# --- END: Added Helper Functions ---


def train_evaluate_model(df: pd.DataFrame, feature_cols: list, target_col: str, model_name: str):
    """Prepares data, trains a RandomForestRegressor, evaluates it, and returns model and preprocessor."""
    logger.info(f"--- Training and evaluating model for target: {target_col} ---")

    if df.empty:
        logger.error(f"Input data for {target_col} model is empty. Skipping.")
        return None, None # Return None for model and preprocessor

    X = df[feature_cols]
    y = df[target_col]

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=int), ['region'])
        ],
        remainder='passthrough' # Keeps 'age'
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=config.RANDOM_STATE)
    logger.info(f"Split data for {target_col}: Train={len(X_train)}, Test={len(X_test)}")

    logger.info("Fitting preprocessor and transforming data...")
    try:
        X_train_processed = preprocessor.fit_transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        ohe_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(['region'])
        passthrough_features = [col for col in feature_cols if col not in ['region']]
        processed_feature_names = np.concatenate([ohe_feature_names, passthrough_features])
        logger.info(f"Processed feature names: {processed_feature_names.tolist()}")
        logger.info(f"Shape of processed training features: {X_train_processed.shape}")
    except Exception as e:
         logger.error(f"Error during preprocessing: {e}", exc_info=True)
         return None, None

    logger.info("Training RandomForestRegressor...")
    rf_model = RandomForestRegressor(n_estimators=100,
                                   random_state=config.RANDOM_STATE,
                                   n_jobs=-1)
    try:
        rf_model.fit(X_train_processed, y_train)
        logger.info("Model training complete.")
    except Exception as e:
        logger.error(f"Error during model training: {e}", exc_info=True)
        return None, None

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
    # Return trained model and fitted preprocessor
    return rf_model, preprocessor


def main():
    """
    Main function: Fetch data, train model, generate predictions, save predictions.
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
    trained_model, fitted_preprocessor = train_evaluate_model(cleaned_df, FEATURE_COLS, TARGET_COL_TRANSACTIONS, "transactions_predictor")

    # Generate predictions on the full cleaned dataset and save to DB
    if trained_model and fitted_preprocessor:
        logger.info(f"Generating predictions for {TARGET_COL_TRANSACTIONS} on full cleaned dataset ({len(cleaned_df)} rows).")
        try:
            # Prepare the full dataset's features using the fitted preprocessor
            X_full_processed = fitted_preprocessor.transform(cleaned_df[FEATURE_COLS])
            full_predictions = trained_model.predict(X_full_processed)

            # Create predictions DataFrame
            predictions_df = pd.DataFrame({
                'customer_id': cleaned_df['customer_id'], # Get customer IDs from the original cleaned data
                'predicted_total_transactions': full_predictions # Keep precision for DB
            })
            logger.info(f"Generated predictions DataFrame, shape: {predictions_df.shape}.")

            # Load predictions to Supabase table
            if REPLACE_EXISTING_PREDICTIONS:
                logger.info(f"REPLACE_EXISTING_PREDICTIONS is True. Clearing table {PREDICTIONS_TABLE_NAME}.")
                if not delete_all_data_from_table(PREDICTIONS_TABLE_NAME, supabase):
                     logger.warning(f"Failed to clear {PREDICTIONS_TABLE_NAME}. Proceeding with insert cautiously.")
                     # Or sys.exit(1)
                else:
                     logger.info(f"Successfully cleared {PREDICTIONS_TABLE_NAME}.")

            if not load_data_to_supabase(predictions_df, PREDICTIONS_TABLE_NAME, supabase):
                logger.error(f"Failed to load predictions into {PREDICTIONS_TABLE_NAME}.")
            else:
                logger.info(f"Successfully loaded predictions into {PREDICTIONS_TABLE_NAME}.")
                # TODO: Update data_history.md for transaction_predictions table population

        except Exception as e_pred_load:
            logger.error(f"Error during prediction generation or loading: {e_pred_load}", exc_info=True)
    else:
        logger.error("Model training failed. Cannot generate or load predictions.")


    logger.info("--- predictive_model.py script execution finished ---")

if __name__ == "__main__":
    main()
```
