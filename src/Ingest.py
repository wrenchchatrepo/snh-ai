# src/Ingest.py
# Description: Extracts customer data from a CSV file and loads it into a Supabase PostgreSQL database.
# Author: Gemini
# Date: 2024-05-17

import os
import sys
import pandas as pd
from supabase import create_client, Client

# Adjust path to import config and snh_logger from parent directory (src)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

try:
    import config
    from src import snh_logger as snh_logging # CORRECTED IMPORT
except ImportError as e:
    print(f"Critical: Could not import 'config' or 'src.snh_logger'. Ensure PYTHONPATH or execution context is correct. Error: {e}")
    import logging # Fallback to standard logging if our custom one fails to import
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler(sys.stderr))
    logger.setLevel(logging.INFO)
    logger.critical(f"Failed to import project modules: {e}. Using basic stderr logger.")
    if "config" in str(e) or "snh_logger" in str(e):
        sys.exit("Exiting: Essential project modules ('config' or 'snh_logger') failed to import.")
else:
    logger = snh_logging.get_logger(__name__)

EXPECTED_COLUMNS = ['customer_id', 'age', 'annual_income', 'total_transactions', 'region']
RAW_TABLE_NAME = 'raw_customer_data'

def extract_data_from_csv(csv_path: str) -> pd.DataFrame | None:
    logger.info(f"Attempting to extract data from CSV: {csv_path}")
    try:
        if not os.path.exists(csv_path):
            logger.error(f"CSV file not found at path: {csv_path}")
            return None
        df = pd.read_csv(csv_path)
        logger.info(f"Successfully loaded CSV. Shape: {df.shape}. Columns: {list(df.columns)}")
        if list(df.columns) != EXPECTED_COLUMNS:
            logger.warning(f"CSV columns {list(df.columns)} do not exactly match expected columns {EXPECTED_COLUMNS}.")
            missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
            if missing_cols:
                logger.error(f"Missing expected columns in CSV: {missing_cols}")
                raise ValueError(f"CSV is missing essential columns: {missing_cols}")
            extra_cols = [col for col in df.columns if col not in EXPECTED_COLUMNS]
            if extra_cols:
                logger.warning(f"CSV has extra columns not in expected list: {extra_cols}")
            logger.info(f"Reordering/selecting columns to match expected schema: {EXPECTED_COLUMNS}")
            df = df[EXPECTED_COLUMNS]
        else:
            logger.info("CSV columns match expected schema and order.")
        if df.empty:
            logger.warning("Loaded CSV data is empty after column validation.")
            return df
        # Convert columns expected to be integers in the DB to nullable integers (Int64)
        # This handles potential NaNs gracefully if they existed and ensures correct type for DB.
        try:
            # If customer_id is always expected to be int, convert it. If it can be text, skip.
            # df['customer_id'] = df['customer_id'].astype('Int64')
            df['age'] = df['age'].astype('Int64')
            df['total_transactions'] = df['total_transactions'].astype('Int64')
            # annual_income is float64, which should map okay to NUMERIC or FLOAT in DB.
            logger.info("Converted 'age' and 'total_transactions' columns to Int64 type.")
        except Exception as e_conv:
             logger.error(f"Error converting column types to Int64: {e_conv}", exc_info=True)
             # Depending on requirements, you might want to return None or raise the error
             return None # Or raise e_conv

        logger.info(f"Data types after conversion:\\n{df.dtypes.to_string()}")
        logger.info(f"First 3 rows of the loaded data after conversion:\\n{df.head(3).to_string()}")
        return df
    except FileNotFoundError:
        logger.error(f"FileNotFoundError: The file {csv_path} was not found.")
        return None
    except pd.errors.EmptyDataError:
        logger.error(f"EmptyDataError: The file {csv_path} is empty (no columns or data).")
        return None
    except pd.errors.ParserError as e:
        logger.error(f"ParserError: Error parsing the file {csv_path}. Details: {e}")
        return None
    except ValueError as e: 
        logger.error(f"ValueError during CSV processing: {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during CSV extraction: {e}", exc_info=True)
        return None

def get_supabase_client() -> Client | None:
    """Initializes and returns a Supabase client."""
    logger.info("Initializing Supabase client.")
    if not config.SUPABASE_URL or not config.SUPABASE_SERVICE_ROLE:
        logger.error("Supabase URL or Service Role Key is not configured. Check .env and config.py.")
        return None
    try:
        supabase_client: Client = create_client(config.SUPABASE_URL, config.SUPABASE_SERVICE_ROLE)
        logger.info("Supabase client initialized successfully.")
        return supabase_client
    except Exception as e:
        logger.error(f"Failed to initialize Supabase client: {e}", exc_info=True)
        return None

def delete_all_data_from_table(table_name: str, supabase_client: Client) -> bool:
    logger.info(f"Attempting to delete all data from Supabase table: {table_name}.")
    try:
        response = supabase_client.table(table_name).delete().neq('customer_id', '__non_existent_value_for_delete_all__').execute()
        if hasattr(response, 'error') and response.error:
            logger.error(f"Error deleting data from Supabase table '{table_name}': {response.error.message}")
            if hasattr(response.error, 'details'): logger.error(f"Details: {response.error.details}")
            if hasattr(response.error, 'hint'): logger.error(f"Hint: {response.error.hint}")
            return False
        deleted_count = len(response.data) if hasattr(response, 'data') and response.data else "an unknown number of"
        logger.info(f"Successfully deleted {deleted_count} records from '{table_name}' (or table was empty). Response indicates success.")
        return True
    except Exception as e:
        logger.error(f"An unexpected error occurred during data deletion from table '{table_name}': {e}", exc_info=True)
        return False

def load_data_to_supabase(df: pd.DataFrame, table_name: str, supabase_client: Client) -> bool:
    if df.empty:
        logger.info(f"DataFrame is empty. No data to load into Supabase table '{table_name}'.")
        return True
    logger.info(f"Attempting to load {len(df)} rows into Supabase table: {table_name}.")
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
            logger.debug(f"Supabase insert response data snippet: {str(response.data)[:200]}")
            return True
        elif not (hasattr(response, 'error') and response.error):
             logger.info(f"Supabase insert for table '{table_name}' completed successfully (no error reported, data may or may not be returned by default).")
             return True
        else: 
            logger.warning(f"Supabase insert for table '{table_name}' resulted in an unexpected response format.")
            logger.debug(f"Full Supabase response: {response}")
            return False
    except Exception as e:
        logger.error(f"An unexpected error occurred during Supabase data loading: {e}", exc_info=True)
        return False

def main():
    logger.info("--- Starting Ingest.py script execution ---")
    REPLACE_EXISTING_DATA = True 
    if not all([config.RAW_DATA_CSV, config.SUPABASE_URL, config.SUPABASE_SERVICE_ROLE_KEY]):
        logger.critical("Essential configurations (RAW_DATA_CSV, SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY) are missing. Aborting.")
        sys.exit(1)
    raw_df = extract_data_from_csv(config.RAW_DATA_CSV)
    if raw_df is None:
        logger.critical("Data extraction from CSV failed. Aborting ingestion process.")
        sys.exit(1)
    if raw_df.empty:
        logger.info("CSV was empty or resulted in an empty DataFrame. No data will be loaded to Supabase.")
    else:
        supabase = get_supabase_client()
        if supabase is None:
            logger.critical("Supabase client initialization failed. Aborting ingestion process.")
            sys.exit(1)
        if REPLACE_EXISTING_DATA:
            logger.info(f"REPLACE_EXISTING_DATA is True. Attempting to delete all data from table '{RAW_TABLE_NAME}'.")
            if not delete_all_data_from_table(RAW_TABLE_NAME, supabase):
                logger.critical(f"Failed to delete data from table '{RAW_TABLE_NAME}'. Aborting to prevent partial load.")
                sys.exit(1)
            else:
                logger.info(f"Successfully cleared data from table '{RAW_TABLE_NAME}'.")
        else:
            logger.info(f"REPLACE_EXISTING_DATA is False. Data will be appended to table '{RAW_TABLE_NAME}'.")
        if not load_data_to_supabase(raw_df, RAW_TABLE_NAME, supabase):
            logger.error(f"Failed to load data into Supabase table '{RAW_TABLE_NAME}'.")
        else:
            logger.info(f"Data ingestion into Supabase table '{RAW_TABLE_NAME}' completed.")
    logger.info("--- Ingest.py script execution finished ---")

if __name__ == "__main__":
    main()