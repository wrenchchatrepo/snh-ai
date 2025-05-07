# src/ml_model.py
# Description: Applies KMeans clustering to identify customer segments.
# Author: Gemini
# Date: 2024-05-07

import os
import sys
import pandas as pd
import numpy as np
from supabase import create_client, Client
from sklearn.cluster import KMeans
# Optional: only if plotting elbow curve directly from script
# import matplotlib.pyplot as plt

# Adjust path to import config and snh_logger from parent directory (src)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

try:
    import config
    from src import snh_logger as snh_logging
except ImportError as e:
    print(f"Critical: Could not import 'config' or 'src.snh_logger'. Ensure PYTHONPATH or execution context is correct. Error: {e}")
    import logging # Fallback to standard logging
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler(sys.stderr))
    logger.setLevel(logging.INFO)
    logger.critical(f"Failed to import project modules: {e}. Using basic stderr logger.")
    if "config" in str(e) or "snh_logger" in str(e):
        sys.exit("Exiting: Essential project modules ('config' or 'snh_logger') failed to import.")
else:
    logger = snh_logging.get_logger(__name__)

TRANSFORMED_TABLE_NAME = 'transformed_customer_data'
SEGMENTS_TABLE_NAME = 'customer_segments'
REPLACE_EXISTING_SEGMENTS = True # Flag to control replacement

# Define columns to be used for clustering
# Exclude customer_id and any other non-feature columns
FEATURE_COLS = [
    'age_scaled', 'annual_income_scaled', 'total_transactions',
    'region_aztec', 'region_celtic', 'region_indus', 'region_nomad'
    # Add 'region_unknown' if it was generated and should be included
]

def get_supabase_client() -> Client | None:
    """Initializes and returns a Supabase client."""
    logger.info("Initializing Supabase client for ml_model.py.")
    if not config.SUPABASE_URL or not config.SUPABASE_SERVICE_ROLE:
        logger.error("Supabase URL or Service Role Key is not configured.")
        return None
    try:
        supabase_client: Client = create_client(config.SUPABASE_URL, config.SUPABASE_SERVICE_ROLE)
        logger.info("Supabase client initialized successfully for ml_model.py.")
        return supabase_client
    except Exception as e:
        logger.error(f"Failed to initialize Supabase client in ml_model.py: {e}", exc_info=True)
        return None

def fetch_transformed_data(supabase_client: Client, table_name: str) -> pd.DataFrame | None:
    """Fetches all data from the specified transformed data table in Supabase."""
    logger.info(f"Attempting to fetch data from Supabase table: {table_name}")
    try:
        response = supabase_client.table(table_name).select("*").execute()
        if hasattr(response, 'error') and response.error:
            logger.error(f"Error fetching data from Supabase table '{table_name}': {response.error.message}")
            return None
        if response.data:
            df = pd.DataFrame(response.data)
            logger.info(f"Successfully fetched {len(df)} rows from '{table_name}'.")
            # Ensure correct numeric types for feature columns
            for col in FEATURE_COLS:
                if col in df.columns:
                    # total_transactions and region columns should be numeric already
                    # scaled columns are float
                     df[col] = pd.to_numeric(df[col], errors='coerce')
                else:
                    logger.error(f"Expected feature column '{col}' not found in fetched data.")
                    return None # Essential column missing
            logger.info(f"Data types after fetch and numeric conversion:\\n{df.dtypes.to_string()}")
            return df
        else:
            logger.info(f"No data found in Supabase table '{table_name}'.")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"An unexpected error occurred while fetching data from '{table_name}': {e}", exc_info=True)
        return None

def find_optimal_k(df: pd.DataFrame, feature_cols: list) -> int:
    """
    Calculates inertia for different values of k and logs them.
    Currently returns a default k, but logs info for manual elbow method analysis.
    """
    if df.empty or not feature_cols:
        logger.error("Cannot find optimal k with empty data or no feature columns.")
        return 0 # Indicate failure

    logger.info("Calculating KMeans inertia for elbow method...")
    X = df[feature_cols].copy()

    # Check for NaNs created during numeric conversion or originally present
    if X.isnull().values.any():
        logger.warning(f"NaN values found in feature columns before clustering. Sum:\\n{X.isnull().sum().to_string()}")
        # Option: Impute NaNs here (e.g., with mean/median) if they shouldn't exist
        # X.fillna(X.median(), inplace=True)
        # logger.info("Filled NaNs with column medians before clustering.")
        # Option: Drop rows with NaNs
        initial_rows = len(X)
        X.dropna(inplace=True)
        logger.warning(f"Dropped {initial_rows - len(X)} rows with NaN values before clustering.")
        if X.empty:
             logger.error("DataFrame became empty after dropping NaN values. Cannot perform clustering.")
             return 0

    inertia = {}
    max_k = config.MAX_CLUSTERS_FOR_ELBOW
    # Ensure we don't try more clusters than samples
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
                            n_init='auto', # Suppress future warning
                            random_state=config.RANDOM_STATE)
            kmeans.fit(X)
            inertia[k] = kmeans.inertia_
            logger.info(f"  k={k}, Inertia={inertia[k]:.2f}")
        except Exception as e:
            logger.error(f"Error calculating KMeans for k={k}: {e}", exc_info=True)
            inertia[k] = None # Mark as failed

    # --- Elbow Method Analysis ---
    # In a real scenario, analyze the logged inertia values (or plot them)
    # to find the "elbow" point where the rate of decrease sharply changes.
    # For this example, we'll just log and return a default/placeholder value.
    logger.info("Elbow method inertia calculation complete. Analyze logged values to determine optimal k.")
    # Optional: Plotting code (requires matplotlib) could be added here to save a plot.

    # --- Determine Optimal K ---
    # This part currently requires manual intervention based on logs/plot.
    # Let's return a default value for now, e.g., 4.
    # TODO: Implement logic to automatically suggest k from inertia values.
    optimal_k = 6 # Changed based on ratio analysis (lowest ratio before increase)
    logger.warning(f"Using optimal k = {optimal_k} based on ratio analysis. ANALYZE INERTIA LOGS/PLOT TO CONFIRM.")

    # Ensure optimal_k is valid before returning
    if optimal_k not in k_range:
         logger.error(f"Selected optimal k ({optimal_k}) is outside the tested/valid range {list(k_range)}. Defaulting to {min(k_range)}.")
         optimal_k = min(k_range) # Fallback to the minimum tested k

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
    original_indices = df.index # Keep original indices for mapping back

    # Handle potential NaNs again before final clustering, just in case
    if X.isnull().values.any():
        logger.warning("NaN values found in feature columns before final clustering. Dropping rows.")
        rows_before_drop = len(X)
        X.dropna(inplace=True)
        if X.empty:
             logger.error("DataFrame became empty after dropping NaN values. Cannot perform final clustering.")
             return None
        # Keep track of indices that remain after dropping NaNs
        valid_indices = X.index
        logger.warning(f"Dropped {rows_before_drop - len(X)} rows with NaN values before final clustering.")
    else:
        valid_indices = original_indices # Use all original indices if no NaNs dropped

    try:
        kmeans = KMeans(n_clusters=optimal_k,
                        init='k-means++',
                        n_init='auto',
                        random_state=config.RANDOM_STATE)
        # Fit only on the valid (non-NaN) feature data
        kmeans.fit(X) 
        # Predict cluster labels for the same valid data points
        labels = kmeans.predict(X) 
        logger.info(f"Successfully fitted KMeans and obtained cluster labels. Cluster centers shape: {kmeans.cluster_centers_.shape}")

        # Create result DataFrame using only the valid indices
        results_df = pd.DataFrame({
            'customer_id': df.loc[valid_indices, 'customer_id'],
            'pattern_id': labels
        })
        logger.info(f"Created results DataFrame with 'customer_id' and 'pattern_id'. Shape: {results_df.shape}")
        return results_df

    except Exception as e:
        logger.error(f"An error occurred during final KMeans fitting or label assignment: {e}", exc_info=True)
        return None


def delete_all_data_from_table(table_name: str, supabase_client: Client) -> bool:
    """Deletes all data from the specified Supabase table."""
    logger.info(f"Attempting to delete all data from Supabase table: {table_name}.")
    try:
        # Assuming 'customer_id' exists and is the PK for the segments table too
        response = supabase_client.table(table_name).delete().neq('customer_id', '__non_existent_value_for_delete_all__').execute()
        if hasattr(response, 'error') and response.error:
            logger.error(f"Error deleting data from Supabase table '{table_name}': {response.error.message}")
            return False
        deleted_count = len(response.data) if hasattr(response, 'data') and response.data else "an unknown number of"
        logger.info(f"Successfully deleted {deleted_count} records from '{table_name}' (or table was empty).")
        return True
    except Exception as e:
        logger.error(f"An unexpected error occurred during data deletion from table '{table_name}': {e}", exc_info=True)
        return False

def load_data_to_supabase(df: pd.DataFrame, table_name: str, supabase_client: Client) -> bool:
    """Loads data from a pandas DataFrame into the specified Supabase table."""
    if df.empty:
        logger.info(f"Results DataFrame is empty. No data to load into Supabase table '{table_name}'.")
        return True

    logger.info(f"Attempting to load {len(df)} rows into Supabase table: {table_name}.")
    # Ensure pattern_id is standard int for JSON/DB
    if 'pattern_id' in df.columns:
        df['pattern_id'] = df['pattern_id'].astype(int) 
        
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
            return False
    except Exception as e:
        logger.error(f"An unexpected error occurred during Supabase data loading for '{table_name}': {e}", exc_info=True)
        return False

def main():
    """
    Main function to orchestrate clustering:
    1. Initialize Supabase client.
    2. Fetch transformed data.
    3. Find optimal k (elbow method - log values).
    4. Assign cluster labels (pattern_id).
    5. Load results (customer_id, pattern_id) to customer_segments table.
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
             
        # Find optimal k (currently logs inertia and returns a placeholder)
        optimal_k = find_optimal_k(transformed_df, actual_feature_cols)
        if optimal_k == 0: # Check if optimal_k calculation failed
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