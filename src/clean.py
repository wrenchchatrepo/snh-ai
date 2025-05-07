# src/clean.py
# Description: Cleans raw customer data by handling missing values and removing duplicates.
# Author: Gemini
# Date: 2024-05-07

import os
import sys
import pandas as pd
import numpy as np
from supabase import create_client, Client

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

RAW_TABLE_NAME = 'raw_customer_data'
CLEANED_TABLE_NAME = 'cleaned_customer_data'
REPLACE_EXISTING_CLEANED_DATA = True # Flag to control if existing data in cleaned table is replaced

def get_supabase_client() -> Client | None:
    """Initializes and returns a Supabase client."""
    logger.info("Initializing Supabase client for clean.py.")
    if not config.SUPABASE_URL or not config.SUPABASE_SERVICE_ROLE_KEY:
        logger.error("Supabase URL or Service Role Key is not configured. Check .env and config.py.")
        return None
    try:
        supabase_client: Client = create_client(config.SUPABASE_URL, config.SUPABASE_SERVICE_ROLE_KEY)
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
            # Supabase might return ints as floats if there were NaNs; Ingest.py converts to Int64.
            # Let's ensure correct types here for cleaning, especially for median calculation.
            if 'age' in df.columns:
                df['age'] = pd.to_numeric(df['age'], errors='coerce')
            if 'annual_income' in df.columns:
                df['annual_income'] = pd.to_numeric(df['annual_income'], errors='coerce')
            if 'total_transactions' in df.columns:
                df['total_transactions'] = pd.to_numeric(df['total_transactions'], errors='coerce')

            logger.info(f"Data types after initial fetch and numeric conversion:\\n{df.dtypes.to_string()}")
            return df
        else:
            logger.info(f"No data found in Supabase table '{table_name}'.")
            # Return an empty DataFrame with expected columns if possible, or handle appropriately.
            # For now, returning None to indicate no data or an issue if data was expected.
            return pd.DataFrame() # Return empty DF, let caller handle
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
    numerical_cols_to_fill = ['age', 'annual_income'] # total_transactions might also be, but usually not NaN
    for col in numerical_cols_to_fill:
        if col in df.columns and df[col].isnull().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            logger.info(f"Filled missing values in '{col}' with median: {median_val:.2f}")
    
    # For 'total_transactions', if it can have NaNs and should be int, fill with 0 or median. Assuming median for consistency.
    if 'total_transactions' in df.columns and df['total_transactions'].isnull().any():
        median_transactions = df['total_transactions'].median()
        df['total_transactions'].fillna(median_transactions, inplace=True)
        logger.info(f"Filled missing values in 'total_transactions' with median: {median_transactions:.0f}")


    # For categorical columns: fill with mode or "Unknown"
    # Let's check 'region'
    if 'region' in df.columns and df['region'].isnull().any():
        # Option 1: Fill with mode
        # mode_val = df['region'].mode()[0] if not df['region'].mode().empty else "Unknown"
        # df['region'].fillna(mode_val, inplace=True)
        # logger.info(f"Filled missing values in 'region' with mode: {mode_val}")
        # Option 2: Fill with "Unknown"
        df['region'].fillna("Unknown", inplace=True)
        logger.info("Filled missing values in 'region' with 'Unknown'.")

    logger.info(f"Missing values after handling:\\n{df.isnull().sum().to_string()}")

    # Convert appropriate columns to nullable Int64 after NaNs are handled
    # This ensures integer columns are truly integers for DB loading.
    int_cols = ['age', 'total_transactions'] # customer_id might already be text or int from DB
    for col in int_cols:
        if col in df.columns:
            try:
                # If NaNs were filled with a float (median), direct astype(Int64) is fine.
                df[col] = df[col].astype('Int64')
                logger.info(f"Converted column '{col}' to Int64 after handling NaNs.")
            except Exception as e_conv:
                logger.warning(f"Could not convert column '{col}' to Int64: {e_conv}")


    # 2. Remove Duplicate Records
    # Considering all columns for identifying duplicates.
    # If specific columns define uniqueness (e.g. customer_id), use subset=['customer_id']
    if df.duplicated().any():
        num_duplicates = df.duplicated().sum()
        df.drop_duplicates(inplace=True, keep='first')
        logger.info(f"Removed {num_duplicates} duplicate records. Shape after duplicate removal: {df.shape}")
    else:
        logger.info("No duplicate records found.")

    logger.info(f"Data cleaning finished. Final shape: {df.shape}. Rows removed: {original_rows - len(df)}")
    logger.info(f"Data types after cleaning:\\n{df.dtypes.to_string()}")
    return df

def delete_all_data_from_table(table_name: str, supabase_client: Client) -> bool:
    """Deletes all data from the specified Supabase table."""
    logger.info(f"Attempting to delete all data from Supabase table: {table_name}.")
    try:
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
        logger.info(f"Cleaned DataFrame is empty. No data to load into Supabase table '{table_name}'.")
        return True

    logger.info(f"Attempting to load {len(df)} rows into Supabase table: {table_name}.")
    records = df.astype(object).where(pd.notnull(df), None).to_dict(orient='records')
    try:
        response = supabase_client.table(table_name).insert(records).execute()
        if hasattr(response, 'error') and response.error:
            logger.error(f"Error inserting data into Supabase table '{table_name}': {response.error.message}")
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
    Main function to orchestrate data cleaning:
    1. Initialize Supabase client.
    2. Fetch raw data from Supabase.
    3. Clean the data.
    4. Load cleaned data into a new Supabase table.
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
        cleaned_df = clean_data(raw_df.copy()) # Use .copy() to avoid SettingWithCopyWarning on the original df

        if cleaned_df.empty and not raw_df.empty : # If cleaning resulted in an empty df from non-empty raw
             logger.warning("Cleaning process resulted in an empty DataFrame. No data will be loaded to cleaned table.")
        elif not cleaned_df.empty:
            if REPLACE_EXISTING_CLEANED_DATA:
                logger.info(f"REPLACE_EXISTING_CLEANED_DATA is True. Attempting to delete all data from table '{CLEANED_TABLE_NAME}'.")
                if not delete_all_data_from_table(CLEANED_TABLE_NAME, supabase):
                    logger.warning(f"Failed to delete all data from '{CLEANED_TABLE_NAME}'. Proceeding with insert, but table may contain old data if it existed.")
                    # Or sys.exit(1) if strict clear is required
                else:
                    logger.info(f"Successfully cleared data from table '{CLEANED_TABLE_NAME}'.")

            # Drop the ingested_at column as it's not in the cleaned table schema
            logger.info(f"Dropping 'ingested_at' column before loading to '{CLEANED_TABLE_NAME}'.")
            cleaned_df.drop(columns=['ingested_at'], inplace=True, errors='ignore') # errors='ignore' prevents error if column missing

            if not load_data_to_supabase(cleaned_df, CLEANED_TABLE_NAME, supabase):
                logger.error(f"Failed to load cleaned data into Supabase table '{CLEANED_TABLE_NAME}'.")
            else:
                logger.info(f"Cleaned data successfully loaded into Supabase table '{CLEANED_TABLE_NAME}'.")
                # TODO: Update data_history.md with info about cleaned_customer_data table & record count.

    logger.info("--- clean.py script execution finished ---")

if __name__ == "__main__":
    main()