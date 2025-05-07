```python
# src/Ingest.py
# Description: Extracts customer data from a CSV file and loads it into a Supabase PostgreSQL database.
# Author: Gemini
# Date: 2024-05-17

import os
import sys
import pandas as pd
from supabase import create_client, Client

# Adjust path to import config and logging from parent directory (src)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

try:
    import config
    from src import logging as snh_logging
except ImportError as e:
    # This fallback might be needed if script is run in certain ways,
    # but ideally, the pipeline execution ensures correct PYTHONPATH.
    print(f"Critical: Could not import 'config' or 'src.logging'. Ensure PYTHONPATH or execution context is correct. Error: {e}")
    # A simple logger if snh_logging fails, writing to stderr
    import logging
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler(sys.stderr))
    logger.setLevel(logging.INFO)
    logger.critical(f"Failed to import project modules: {e}. Using basic stderr logger.")
    if "config" in str(e):
        sys.exit("Exiting: 'config' module is essential.")
else:
    logger = snh_logging.get_logger(__name__)

EXPECTED_COLUMNS = ['customer_id', 'age', 'annual_income', 'total_transactions', 'region']
RAW_TABLE_NAME = 'raw_customer_data' # As per data_history.md

def extract_data_from_csv(csv_path: str) -> pd.DataFrame | None:
    """
    Extracts data from the specified CSV file into a pandas DataFrame.
    Performs basic validation on column names.
    """
    logger.info(f"Attempting to extract data from CSV: {csv_path}")
    try:
        if not os.path.exists(csv_path):
            logger.error(f"CSV file not found at path: {csv_path}")
            return None
        
        df = pd.read_csv(csv_path)
        logger.info(f"Successfully loaded CSV. Shape: {df.shape}. Columns: {list(df.columns)}")

        if list(df.columns) != EXPECTED_COLUMNS:
            logger.warning(f"CSV columns {list(df.columns)} do not exactly match expected columns {EXPECTED_COLUMNS}.")
            # For robustness, check if all expected columns are present, even if order or extra columns exist
            missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
            if missing_cols:
                logger.error(f"Missing expected columns in CSV: {missing_cols}")
                raise ValueError(f"CSV is missing essential columns: {missing_cols}")
            extra_cols = [col for col in df.columns if col not in EXPECTED_COLUMNS]
            if extra_cols:
                logger.warning(f"CSV has extra columns not in expected list: {extra_cols}")
            # If all expected columns are present, reorder/select to match expected schema
            logger.info(f"Reordering/selecting columns to match expected schema: {EXPECTED_COLUMNS}")
            df = df[EXPECTED_COLUMNS]
        else:
            logger.info("CSV columns match expected schema and order.")

        if df.empty:
            logger.warning("Loaded CSV data is empty after column validation.")
            # Return empty DataFrame, let downstream process decide how to handle.
            return df

        logger.info(f"Data types from CSV:\n{df.dtypes.to_string()}")
        logger.info(f"First 3 rows of the loaded data:\n{df.head(3).to_string()}")
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
    except ValueError as e: # Catch our specific column validation error
        logger.error(f"ValueError during CSV processing: {e}")
        return None
    except Exception as e:
        logger.error(f"An unexpected error occurred during CSV extraction: {e}", exc_info=True)
        return None

def get_supabase_client() -> Client | None:
    """Initializes and returns a Supabase client."""
    logger.info("Initializing Supabase client.")
    if not config.SUPABASE_URL or not config.SUPABASE_SERVICE_ROLE_KEY:
        logger.error("Supabase URL or Service Role Key is not configured. Check .env and config.py.")
        return None
    try:
        supabase_client: Client = create_client(config.SUPABASE_URL, config.SUPABASE_SERVICE_ROLE_KEY)
        logger.info("Supabase client initialized successfully.")
        return supabase_client
    except Exception as e:
        logger.error(f\"Failed to initialize Supabase client: {e}\", exc_info=True)
        return None

def delete_all_data_from_table(table_name: str, supabase_client: Client) -> bool:
    """
    Deletes all data from the specified Supabase table.
    Uses a non-matching inequality condition as a common way to achieve 'delete all'
    if the client library requires a filter for delete operations.
    Adjust the column name if 'customer_id' is not suitable or a different 'always true'
    or 'always false for neq' filter is preferred.
    """
    logger.info(f\"Attempting to delete all data from Supabase table: {table_name}.\")
    try:
        # A common pattern for deleting all rows if a filter is mandatory.
        # Using a condition that is unlikely to be met by 'neq' effectively targets all rows.
        # Alternatively, if your table has a boolean 'is_active' or similar,
        # you could use .delete().eq('is_active', True).Or('is_active', False)
        # Or if the primary key is an integer: .delete().gte(primary_key_column, 0)
        # For this example, we use neq with a placeholder value.
        # Ensure 'customer_id' is a valid column in your table. If not, use another, or check Supabase client docs for filterless delete.
        response = supabase_client.table(table_name).delete().neq('customer_id', '__non_existent_value_for_delete_all__').execute()

        if hasattr(response, 'error') and response.error:
            logger.error(f\"Error deleting data from Supabase table \'{table_name}\': {response.error.message}\")
            if hasattr(response.error, 'details'): logger.error(f\"Details: {response.error.details}\")
            if hasattr(response.error, 'hint'): logger.error(f\"Hint: {response.error.hint}\")
            return False
        
        # Successful deletion usually results in data being an empty list or a count of deleted rows,
        # depending on the server/client version and 'returning' preferences.
        # For a delete-all, confirming no error is the primary success indicator.
        deleted_count = len(response.data) if hasattr(response, 'data') and response.data else "an unknown number of"
        logger.info(f\"Successfully deleted {deleted_count} records from \'{table_name}\' (or table was empty). Response indicates success.\")
        return True

    except Exception as e:
        logger.error(f\"An unexpected error occurred during data deletion from table \'{table_name}\': {e}\", exc_info=True)
        return False

def load_data_to_supabase(df: pd.DataFrame, table_name: str, supabase_client: Client) -> bool:
    """
    Loads data from a pandas DataFrame into the specified Supabase table.
    The table is expected to exist.
    """
    if df.empty:
        logger.info(f"DataFrame is empty. No data to load into Supabase table '{table_name}'.")
        return True

    logger.info(f"Attempting to load {len(df)} rows into Supabase table: {table_name}.")
    
    # Convert DataFrame to list of dictionaries.
    # pd.NA is preferred for missing values, but Supabase client might handle None better.
    # Let's ensure NaNs become None for JSON serialization compatibility.
    records = df.astype(object).where(pd.notnull(df), None).to_dict(orient='records')

    try:
        # The supabase-py client insert method sends all rows in a single request.
        # For very large datasets, chunking might be needed if API limits are hit.
        # This is a simple insert for now.
        response = supabase_client.table(table_name).insert(records).execute()

        # Error handling based on supabase-py v1.x typical response structure
        if hasattr(response, 'error') and response.error:
            logger.error(f"Error inserting data into Supabase table '{table_name}': {response.error.message}")
            if hasattr(response.error, 'details'): logger.error(f"Details: {response.error.details}")
            if hasattr(response.error, 'hint'): logger.error(f"Hint: {response.error.hint}")
            return False
        elif hasattr(response, 'data') and response.data:
            logger.info(f"Successfully inserted {len(response.data)} records into '{table_name}'.")
            logger.debug(f\"Supabase insert response data snippet: {str(response.data)[:200]}\")
            return True
        elif not (hasattr(response, 'error') and response.error):
            # If there's no error attribute, or it's present but falsy (None, empty dict),
            # and data might be empty (e.g., insert without returning='representation'),
            # consider it a success.
            logger.info(f\"Supabase insert for table \'{table_name}\' completed successfully (no error reported, data may or may not be returned by default).\")
            return True
        else:
            # This case now more definitively means an unexpected response.
            logger.warning(f\"Supabase insert for table \'{table_name}\' resulted in an unexpected response format.\")
            logger.debug(f\"Full Supabase response: {response}\")
            return False

    except Exception as e:
        logger.error(f\"An unexpected error occurred during Supabase data loading: {e}\", exc_info=True)
        return False

def main():
    """
    Main function to orchestrate data ingestion:
    1. Extract data from CSV.
    2. Initialize Supabase client.
    3. Load extracted data into Supabase.
    """
    logger.info("--- Starting Ingest.py script execution ---")

    # Configuration for replacing data
    REPLACE_EXISTING_DATA = True # Set to False to append, True to delete existing before insert

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
            logger.info(f\"REPLACE_EXISTING_DATA is True. Attempting to delete all data from table \'{RAW_TABLE_NAME}\'.\")
            if not delete_all_data_from_table(RAW_TABLE_NAME, supabase):
                logger.critical(f\"Failed to delete data from table \'{RAW_TABLE_NAME}\'. Aborting to prevent partial load.\")
                sys.exit(1)
            else:
                logger.info(f\"Successfully cleared data from table \'{RAW_TABLE_NAME}\'.\")
        else:
            logger.info(f\"REPLACE_EXISTING_DATA is False. Data will be appended to table \'{RAW_TABLE_NAME}\'.\")

        if not load_data_to_supabase(raw_df, RAW_TABLE_NAME, supabase):
            logger.error(f\"Failed to load data into Supabase table \'{RAW_TABLE_NAME}\'.\")
        else:
            logger.info(f\"Data ingestion into Supabase table \'{RAW_TABLE_NAME}\' completed.\")

    logger.info("--- Ingest.py script execution finished ---")

if __name__ == "__main__":
    main()
```