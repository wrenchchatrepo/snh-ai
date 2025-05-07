# src/ml_model.py
# Description: Applies KMeans clustering to identify customer segments.
# Date: 2024-05-07

import os
import sys
import pandas as pd
import numpy as np
import traceback
from supabase import create_client, Client
from sklearn.cluster import KMeans
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

TRANSFORMED_TABLE_NAME = 'transformed_customer_data'
SEGMENTS_TABLE_NAME = 'customer_segments'
REPLACE_EXISTING_SEGMENTS = True # Flag to control replacement
MODEL_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "models") # Directory to save trained models

# Define columns to be used for clustering
FEATURE_COLS = [
    'age_scaled', 'annual_income_scaled', 'total_transactions',
    'region_aztec', 'region_celtic', 'region_indus', 'region_nomad'
    # Add 'region_unknown' if needed based on transform.py output/DDL
]

def get_supabase_client() -> Client | None:
    """Initializes and returns a Supabase client."""
    logger.info("Initializing Supabase client for ml_model.py.")
    if not config.SUPABASE_URL or not config.SUPABASE_SERVICE_ROLE:
        logger.error("Supabase URL or Service Role is not configured.")
        return None
    try:
        supabase_client: Client = create_client(config.SUPABASE_URL, config.SUPABASE_SERVICE_ROLE)
        logger.info("Supabase client initialized successfully for ml_model.py.")
        return supabase_client
    except Exception as e:
        logger.error(f"Failed to initialize Supabase client in ml_model.py: {e}", exc_info=True)
        return None

def fetch_transformed_data(supabase_client: Client, table_name: str) -> pd.DataFrame | None:
    """Fetches feature data from the specified transformed data table."""
    logger.info(f"Attempting to fetch data from Supabase table: {table_name}")
    try:
        select_columns = ["customer_id"] + FEATURE_COLS
        select_query = ", ".join(select_columns)
        logger.info(f"Selecting columns: {select_query}")

        response = supabase_client.table(table_name).select(select_query).execute()

        if hasattr(response, 'error') and response.error:
            logger.error(f"Error fetching data from Supabase table '{table_name}': {response.error.message}")
            return None
        if response.data:
            df = pd.DataFrame(response.data)
            logger.info(f"Successfully fetched {len(df)} rows from '{table_name}'.")
            # Ensure correct numeric types for feature columns
            all_cols_found = True
            for col in FEATURE_COLS:
                if col in df.columns:
                     df[col] = pd.to_numeric(df[col], errors='coerce')
                else:
                    logger.error(f"Expected feature column '{col}' not found in fetched data.")
                    all_cols_found = False
            if 'customer_id' not in df.columns:
                 logger.error("Critical column 'customer_id' not found in fetched data.")
                 all_cols_found = False

            if not all_cols_found:
                 return None

            # Check for NaNs introduced by coercion
            if df[FEATURE_COLS].isnull().values.any():
                logger.warning(f"NaN values found in feature columns after fetch/coerce. Sum:\\n{df[FEATURE_COLS].isnull().sum().to_string()}")
                initial_rows = len(df)
                df.dropna(subset=FEATURE_COLS, inplace=True)
                logger.warning(f"Dropped {initial_rows - len(df)} rows containing NaN in feature columns.")
                if df.empty:
                     logger.error("DataFrame became empty after dropping NaN values from features.")
                     return pd.DataFrame()

            logger.info(f"Data types after fetch and numeric conversion:\\n{df.dtypes.to_string()}")
            # Ensure customer_id is string type for consistency
            df['customer_id'] = df['customer_id'].astype(str)
            return df
        else:
            logger.info(f"No data found in Supabase table '{table_name}'.")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"An unexpected error occurred while fetching data from '{table_name}': {e}", exc_info=True)
        return None


def find_optimal_k(df: pd.DataFrame, feature_cols: list) -> int:
    """
    Calculates KMeans SSE (inertia) for different values of k and logs them.
    Returns k=6 based on previous analysis, logs info for validation.
    """
    if df.empty or not feature_cols:
        logger.error("Cannot find optimal k with empty data or no feature columns.")
        return 0

    logger.info("Calculating KMeans SSE (inertia) for elbow method...")
    X = df[feature_cols].copy()

    if X.isnull().values.any():
        logger.error("NaN values still present in features before clustering. Aborting k search.")
        return 0

    inertia = {}
    max_k = config.MAX_CLUSTERS_FOR_ELBOW
    effective_max_k = min(max_k, len(X))
    if effective_max_k < 2:
        logger.error(f"Not enough samples ({len(X)}) to test at least 2 clusters.")
        return 0

    k_range = range(2, effective_max_k + 1)

    logger.info(f"Testing k from {min(k_range)} to {effective_max_k}...")
    for k in k_range:
        try:
            kmeans = KMeans(n_clusters=k,
                            init='k-means++',
                            n_init='auto',
                            random_state=config.RANDOM_STATE)
            kmeans.fit(X)
            inertia[k] = kmeans.inertia_
            logger.info(f"  k={k}, SSE (Inertia)={inertia[k]:.2f}")
        except Exception as e:
            logger.error(f"Error calculating KMeans for k={k}: {e}", exc_info=True)
            inertia[k] = None

    logger.info("Elbow method SSE calculation complete. Review logged values.")

    optimal_k = 6 # Set based on previous ratio analysis
    logger.warning(f"Using optimal k = {optimal_k} based on prior analysis. Review SSE/elbow plot if needed.")

    if optimal_k not in k_range:
         logger.error(f"Selected optimal k ({optimal_k}) is outside the tested/valid range {list(k_range)}. Defaulting to {min(k_range)}.")
         optimal_k = min(k_range)

    return optimal_k

def assign_clusters(df: pd.DataFrame, feature_cols: list, optimal_k: int) -> pd.DataFrame | None:
    """
    Fits KMeans with the optimal k and assigns cluster labels ('pattern_id').
    Returns a DataFrame with 'customer_id' and 'pattern_id'.
    """
    if df.empty or not feature_cols or optimal_k < 2:
        logger.error("Cannot assign clusters with empty data, no features, or invalid k.")
        return None

    logger.info(f"Fitting final KMeans model with k={optimal_k}...")
    X = df[feature_cols].copy()

    if X.isnull().values.any():
        logger.error("NaN values detected unexpectedly before final clustering. Aborting.")
        return None

    try:
        kmeans = KMeans(n_clusters=optimal_k,
                        init='k-means++',
                        n_init='auto',
                        random_state=config.RANDOM_STATE)
        labels = kmeans.fit_predict(X)
        logger.info(f"Successfully fitted KMeans and obtained cluster labels. Cluster centers shape: {kmeans.cluster_centers_.shape}")

        results_df = pd.DataFrame({
            'customer_id': df['customer_id'], # Use original df customer_id
            'pattern_id': labels
        })
        logger.info(f"Created results DataFrame with 'customer_id' and 'pattern_id'. Shape: {results_df.shape}")

        # Optional: Save the fitted KMeans model
        os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
        model_filename = os.path.join(MODEL_OUTPUT_DIR, f"kmeans_k{optimal_k}_model.joblib")
        try:
            joblib.dump(kmeans, model_filename)
            logger.info(f"Saved KMeans model to {model_filename}")
        except Exception as e_save:
            logger.error(f"Error saving KMeans model: {e_save}", exc_info=True)

        return results_df

    except Exception as e:
        logger.error(f"An error occurred during final KMeans fitting or label assignment: {e}", exc_info=True)
        return None


def delete_all_data_from_table(table_name: str, supabase_client: Client) -> bool:
    """Deletes all data from the specified Supabase table."""
    logger.info(f"Attempting to delete all data from Supabase table: {table_name}.")
    try:
        response = supabase_client.table(table_name).delete().neq('customer_id', '__non_existent_value_for_delete_all__').execute()
        if hasattr(response, 'error') and response.error:
            if 'does not exist' in response.error.message:
                 logger.warning(f"Table '{table_name}' does not seem to exist. Skipping delete.")
                 return True
            logger.error(f"Error deleting data from Supabase table '{table_name}': {response.error.message}")
            return False
        deleted_count = len(response.data) if hasattr(response, 'data') and response.data else "an unknown number of"
        logger.info(f"Successfully deleted {deleted_count} records from '{table_name}' (or table was empty/non-existent).")
        return True
    except Exception as e:
        logger.error(f"An unexpected error occurred during data deletion from table '{table_name}': {e}", exc_info=True)
        return False

def load_data_to_supabase(df: pd.DataFrame, table_name: str, supabase_client: Client) -> bool:
    """Loads segment data from a pandas DataFrame into the specified Supabase table."""
    if df.empty:
        logger.info(f"Results DataFrame is empty. No data to load into Supabase table '{table_name}'.")
        return True

    logger.info(f"Attempting to load {len(df)} rows into Supabase table: {table_name}.")

    if 'pattern_id' in df.columns:
        df['pattern_id'] = df['pattern_id'].astype(int)
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
        elif hasattr(response, 'data') and response.data:
            logger.info(f"Successfully inserted {len(response.data)} records into '{table_name}'.")
            return True
        elif not (hasattr(response, 'error') and response.error):
             logger.info(f"Supabase insert for table '{table_name}' completed successfully (no error reported).")
             return True
        else:
            logger.warning(f"Supabase insert for table '{table_name}' resulted in an unexpected response format.")
            logger.debug(f"Full Supabase response: {response}")
            return False
    except Exception as e:
        logger.error(f"An unexpected error occurred during Supabase data loading for '{table_name}': {e}", exc_info=True)
        return False


def main():
    """
    Main function to orchestrate clustering: Fetch, Find k, Cluster, Load results.
    """
    logger.info("--- Starting ml_model.py script execution ---")

    supabase = get_supabase_client()
    if supabase is None:
        logger.critical("Supabase client initialization failed in ml_model.py. Aborting.")
        sys.exit(1)

    transformed_df = fetch_transformed_data(supabase, TRANSFORMED_TABLE_NAME)
    if transformed_df is None:
        logger.critical(f"Failed to fetch transformed data from '{TRANSFORMED_TABLE_NAME}'. Aborting ML process.")
        sys.exit(1)

    if transformed_df.empty:
        logger.info(f"No data fetched from '{TRANSFORMED_TABLE_NAME}'. Cannot perform clustering.")
    else:
        # Ensure feature columns actually exist in the fetched data
        actual_feature_cols = [col for col in FEATURE_COLS if col in transformed_df.columns]
        if len(actual_feature_cols) != len(FEATURE_COLS):
             missing_features = list(set(FEATURE_COLS) - set(actual_feature_cols))
             logger.error(f"Missing expected feature columns in fetched data: {missing_features}. Aborting.")
             sys.exit(1)
        if not actual_feature_cols:
             logger.error("No feature columns available for clustering. Aborting.")
             sys.exit(1)

        # Find optimal k
        optimal_k = find_optimal_k(transformed_df, actual_feature_cols)
        if optimal_k == 0:
             logger.error("Failed to determine optimal k. Aborting.")
             sys.exit(1)

        # Assign cluster labels
        segment_results_df = assign_clusters(transformed_df, actual_feature_cols, optimal_k)

        if segment_results_df is None:
             logger.error("Failed to assign cluster labels. Aborting loading process.")
             sys.exit(1)
        elif segment_results_df.empty:
             logger.warning("Segment results DataFrame is empty. No data will be loaded.")
        else:
            # Load results to the final table
            if REPLACE_EXISTING_SEGMENTS:
                logger.info(f"REPLACE_EXISTING_SEGMENTS is True. Attempting to delete all data from table '{SEGMENTS_TABLE_NAME}'.")
                if not delete_all_data_from_table(SEGMENTS_TABLE_NAME, supabase):
                    logger.warning(f"Failed to delete data from '{SEGMENTS_TABLE_NAME}'. Proceeding with insert, but table may contain old data.")
                else:
                    logger.info(f"Successfully cleared data from table '{SEGMENTS_TABLE_NAME}'.")

            if not load_data_to_supabase(segment_results_df, SEGMENTS_TABLE_NAME, supabase):
                logger.error(f"Failed to load segment results into Supabase table '{SEGMENTS_TABLE_NAME}'.")
            else:
                logger.info(f"Segment results successfully loaded into Supabase table '{SEGMENTS_TABLE_NAME}'.")
                # TODO: Update data_history.md automatically or prompt user.

    logger.info("--- ml_model.py script execution finished ---")

if __name__ == "__main__":
    main()
