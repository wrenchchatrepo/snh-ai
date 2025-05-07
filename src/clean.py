# src/clean.py
# Description: Cleans raw customer data by handling missing values and removing duplicates.
# Date: 2024-05-07

import os
import sys
import pandas as pd
import numpy as np
import traceback
from supabase import create_client, Client

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

RAW_TABLE_NAME = 'raw_customer_data'
CLEANED_TABLE_NAME = 'cleaned_customer_data'
REPLACE_EXISTING_CLEANED_DATA = True # Flag to control if existing data in cleaned table is replaced

def get_supabase_client() -> Client | None:
    """Initializes and returns a Supabase client."""
    logger.info("Initializing Supabase client for clean.py.")
    # Use the corrected variable name config.SUPABASE_SERVICE_ROLE
    if not config.SUPABASE_URL or not config.SUPABASE_SERVICE_ROLE:
        logger.error("Supabase URL or Service Role is not configured. Check .env and config.py.")
        return None
    try:
         # Use the corrected variable name config.SUPABASE_SERVICE_ROLE
        supabase_client: Client = create_client(config.SUPABASE_URL, config.SUPABASE_SERVICE_ROLE)
        logger.info("Supabase client initialized successfully for clean.py.")
        return supabase_client
    except Exception as e:
        logger.error(f"Failed to initialize Supabase client in clean.py: {e}", exc_info=True)
        return None

def fetch_raw_data(supabase_client: Client, table_name: str) -> pd.DataFrame | None:
    """Fetches all data from the specified raw data table in Supabase."""
    logger.info(f"Attempting to fetch data from Supabase table: {table_name}")
    try:
        response = supabase_client.table(table_name).select("*").execute()
        if hasattr(response, 'error') and response.error:
            logger.error(f"Error fetching data from Supabase table '{table_name}': {response.error.message}")
            return None
        if response.data:
            df = pd.DataFrame(response.data)
            logger.info(f"Successfully fetched {len(df)} rows from '{table_name}'.")
            # Ensure correct types after fetch
            numeric_cols = ['age', 'annual_income', 'total_transactions']
            for col in numeric_cols:
                 if col in df.columns:
                      # Coerce to numeric, errors become NaN. Use float for flexibility before Int64 conversion.
                      df[col] = pd.to_numeric(df[col], errors='coerce')
            if 'region' in df.columns:
                df['region'] = df['region'].astype(str) # Ensure region is string
            if 'customer_id' in df.columns:
                df['customer_id'] = df['customer_id'].astype(str)

            logger.info(f"Data types after initial fetch and numeric coercion:\\n{df.dtypes.to_string()}")
            return df
        else:
            logger.info(f"No data found in Supabase table '{table_name}'.")
            return pd.DataFrame() # Return empty DF
    except Exception as e:
        logger.error(f"An unexpected error occurred while fetching data from '{table_name}': {e}", exc_info=True)
        return None

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans the DataFrame by handling missing values and removing duplicates."""
    if df.empty:
        logger.info("Input DataFrame is empty. No cleaning to perform.")
        return df

    logger.info(f"Starting data cleaning. Initial shape: {df.shape}")
    original_rows = len(df)

    # 1. Handle Missing Values
    logger.info(f"Missing values before handling:\\n{df.isnull().sum().to_string()}")

    # For numerical columns: fill with median
    numerical_cols_to_fill = ['age', 'annual_income', 'total_transactions']
    for col in numerical_cols_to_fill:
        if col in df.columns and df[col].isnull().any():
            median_val = df[col].dropna().median()
            if pd.isna(median_val):
                 median_val = 0
                 logger.warning(f"Column '{col}' contained only NaN values. Filling missing with 0.")
            df[col] = df[col].fillna(median_val) # Use direct assignment
            logger.info(f"Filled missing values in '{col}' with median: {median_val:.2f}")

    # For categorical columns: fill with "Unknown"
    if 'region' in df.columns and df['region'].isnull().any():
        df['region'] = df['region'].fillna("Unknown") # Use direct assignment
        logger.info("Filled missing values in 'region' with 'Unknown'.")

    logger.info(f"Missing values after handling:\\n{df.isnull().sum().to_string()}")

    # Convert appropriate columns to nullable Int64 after NaNs are handled
    int_cols = ['age', 'total_transactions']
    for col in int_cols:
        if col in df.columns:
            try:
                # Convert float medians back to Int64
                df[col] = df[col].astype(float).astype('Int64')
                logger.info(f"Converted column '{col}' to Int64 after handling NaNs.")
            except Exception as e_conv:
                logger.warning(f"Could not convert column '{col}' to Int64: {e_conv}")

    # Ensure annual_income remains float
    if 'annual_income' in df.columns:
         df['annual_income'] = pd.to_numeric(df['annual_income'], errors='coerce')

    # 2. Remove Duplicate Records
    if df.duplicated().any():
        num_duplicates = df.duplicated().sum()
        df.drop_duplicates(inplace=True, keep='first')
        logger.info(f"Removed {num_duplicates} duplicate records. Shape after duplicate removal: {df.shape}")
    else:
        logger.info("No duplicate records found.")

    logger.info(f"Data cleaning finished. Final shape: {df.shape}. Rows removed: {original_rows - len(df)}")
    logger.info(f"Data types after cleaning:\\n{df.dtypes.to_string()}")
    return df

# --- START: Added Helper Functions ---
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
    """Loads cleaned data from a pandas DataFrame into the specified Supabase table."""
    if df.empty:
        logger.info(f"Cleaned DataFrame is empty. No data to load into Supabase table '{table_name}'.")
        return True

    logger.info(f"Attempting to load {len(df)} rows into Supabase table: {table_name}.")

    # Ensure correct types before conversion to dicts for JSON
    if 'customer_id' in df.columns:
         df['customer_id'] = df['customer_id'].astype(str)
    if 'age' in df.columns:
         df['age'] = df['age'].astype(object).where(pd.notnull(df['age']), None)
    if 'total_transactions' in df.columns:
         df['total_transactions'] = df['total_transactions'].astype(object).where(pd.notnull(df['total_transactions']), None)
    if 'annual_income' in df.columns:
         df['annual_income'] = pd.to_numeric(df['annual_income'], errors='coerce')
         df['annual_income'] = df['annual_income'].astype(object).where(pd.notnull(df['annual_income']), None)
    if 'region' in df.columns:
         df['region'] = df['region'].astype(str)

    records = df.to_dict(orient='records')

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
# --- END: Added Helper Functions ---


def main():
    """
    Main function to orchestrate data cleaning: Fetch, Clean, Load.
    """
    logger.info("--- Starting clean.py script execution ---")

    supabase = get_supabase_client()
    if supabase is None:
        logger.critical("Supabase client initialization failed in clean.py. Aborting.")
        sys.exit(1)

    raw_df = fetch_raw_data(supabase, RAW_TABLE_NAME)
    if raw_df is None:
        logger.critical(f"Failed to fetch raw data from '{RAW_TABLE_NAME}'. Aborting cleaning process.")
        sys.exit(1)

    if raw_df.empty:
        logger.info(f"No data fetched from '{RAW_TABLE_NAME}'. Nothing to clean or load.")
    else:
        cleaned_df = clean_data(raw_df.copy()) # Use .copy() to avoid warnings

        if cleaned_df.empty and not raw_df.empty :
             logger.warning("Cleaning process resulted in an empty DataFrame. No data will be loaded to cleaned table.")
        elif not cleaned_df.empty:
            # Drop columns not in the target table schema before loading
            cols_to_keep = ['customer_id', 'age', 'annual_income', 'total_transactions', 'region']
            cols_to_drop = [col for col in cleaned_df.columns if col not in cols_to_keep]
            if cols_to_drop:
                logger.info(f"Dropping columns not in target schema before loading: {cols_to_drop}")
                cleaned_df.drop(columns=cols_to_drop, inplace=True)

            if REPLACE_EXISTING_CLEANED_DATA:
                logger.info(f"REPLACE_EXISTING_CLEANED_DATA is True. Attempting to delete all data from table '{CLEANED_TABLE_NAME}'.")
                if not delete_all_data_from_table(CLEANED_TABLE_NAME, supabase):
                    logger.warning(f"Failed to delete all data from '{CLEANED_TABLE_NAME}'. Proceeding with insert, but table may contain old data if it existed.")
                else:
                    logger.info(f"Successfully cleared data from table '{CLEANED_TABLE_NAME}'.")

            if not load_data_to_supabase(cleaned_df, CLEANED_TABLE_NAME, supabase):
                logger.error(f"Failed to load cleaned data into Supabase table '{CLEANED_TABLE_NAME}'.")
            else:
                logger.info(f"Cleaned data successfully loaded into Supabase table '{CLEANED_TABLE_NAME}'.")
                # TODO: Update data_history.md

    logger.info("--- clean.py script execution finished ---")

if __name__ == "__main__":
    main()
