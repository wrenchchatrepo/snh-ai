# src/transform.py
# Description: Scales numerical features and encodes categorical features from cleaned data.
# Date: 2024-05-07

import os
import sys
import pandas as pd
import numpy as np
import traceback
from supabase import create_client, Client
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline # Not strictly needed here

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
TRANSFORMED_TABLE_NAME = 'transformed_customer_data'
REPLACE_EXISTING_TRANSFORMED_DATA = True # Flag to control replacement

# Define columns
NUMERICAL_COLS = ['age', 'annual_income']
CATEGORICAL_COLS = ['region']
PASSTHROUGH_COLS = ['customer_id', 'total_transactions'] # Columns to keep as is


def get_supabase_client() -> Client | None:
    """Initializes and returns a Supabase client."""
    logger.info("Initializing Supabase client for transform.py.")
    # Use the corrected variable name from config.py
    if not config.SUPABASE_URL or not config.SUPABASE_SERVICE_ROLE:
        logger.error("Supabase URL or Service Role is not configured.")
        return None
    try:
        # Use the corrected variable name from config.py
        supabase_client: Client = create_client(config.SUPABASE_URL, config.SUPABASE_SERVICE_ROLE)
        logger.info("Supabase client initialized successfully for transform.py.")
        return supabase_client
    except Exception as e:
        logger.error(f"Failed to initialize Supabase client in transform.py: {e}", exc_info=True)
        return None

def fetch_cleaned_data(supabase_client: Client, table_name: str) -> pd.DataFrame | None:
    """Fetches all data from the specified cleaned data table in Supabase."""
    logger.info(f"Attempting to fetch data from Supabase table: {table_name}")
    try:
        # Fetch all columns needed for transformation + passthrough
        select_columns = PASSTHROUGH_COLS + NUMERICAL_COLS + CATEGORICAL_COLS
        # Remove potential duplicates like customer_id if it's in multiple lists
        select_columns = sorted(list(set(select_columns)))
        select_query = ", ".join(select_columns)
        logger.info(f"Selecting columns: {select_query}")

        response = supabase_client.table(table_name).select(select_query).execute()

        if hasattr(response, 'error') and response.error:
            logger.error(f"Error fetching data from Supabase table '{table_name}': {response.error.message}")
            return None
        if response.data:
            df = pd.DataFrame(response.data)
            logger.info(f"Successfully fetched {len(df)} rows from '{table_name}'.")
            # Ensure correct dtypes before transformation
            if 'age' in df.columns:
                df['age'] = pd.to_numeric(df['age'], errors='coerce').astype('Int64')
            if 'annual_income' in df.columns:
                df['annual_income'] = pd.to_numeric(df['annual_income'], errors='coerce')
            if 'total_transactions' in df.columns:
                df['total_transactions'] = pd.to_numeric(df['total_transactions'], errors='coerce').astype('Int64')
            if 'region' in df.columns:
                 df['region'] = df['region'].astype(str) # Ensure region is string
            if 'customer_id' in df.columns:
                 df['customer_id'] = df['customer_id'].astype(str)

            # Handle NaNs - Although clean.py should have handled them, check again
            initial_rows = len(df)
            df.dropna(subset=select_columns, inplace=True) # Drop rows missing any needed value
            if len(df) < initial_rows:
                 logger.warning(f"Dropped {initial_rows - len(df)} rows containing NaN values fetched from {table_name}.")

            if df.empty:
                 logger.error(f"DataFrame is empty after dropping NaN values from {table_name}. Cannot proceed.")
                 return pd.DataFrame()

            logger.info(f"Data types after fetch and type checks:\\n{df.dtypes.to_string()}")
            return df
        else:
            logger.info(f"No data found in Supabase table '{table_name}'.")
            return pd.DataFrame()
    except Exception as e:
        logger.error(f"An unexpected error occurred while fetching data from '{table_name}': {e}", exc_info=True)
        return None

def transform_data(df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Applies scaling to numerical columns and one-hot encoding to categorical columns.
    Returns the transformed DataFrame.
    """
    if df.empty:
        logger.info("Input DataFrame is empty. No transformation to perform.")
        return df

    logger.info(f"Starting data transformation. Initial shape: {df.shape}")

    # Ensure only columns intended for processing are present before transformation
    cols_for_transform = NUMERICAL_COLS + CATEGORICAL_COLS + PASSTHROUGH_COLS
    cols_for_transform = sorted(list(set(cols_for_transform))) # Unique sorted list
    df_processed = df[cols_for_transform].copy() # Work on a copy with only needed cols

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERICAL_COLS),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=int), CATEGORICAL_COLS)
        ],
        remainder='passthrough' # Keeps PASSTHROUGH_COLS ('customer_id', 'total_transactions')
    )

    try:
        transformed_data = preprocessor.fit_transform(df_processed)
        feature_names_out = preprocessor.get_feature_names_out()
        logger.info(f"Feature names after transformation: {feature_names_out}")

        transformed_df = pd.DataFrame(transformed_data, columns=feature_names_out, index=df_processed.index)

        # Rename columns for clarity and DB compatibility
        column_mapping = {}
        db_column_mapping = { # For case adjustment
            'region_Aztec': 'region_aztec',
            'region_Celtic': 'region_celtic',
            'region_Indus': 'region_indus',
            'region_Nomad': 'region_nomad',
            'region_Unknown': 'region_unknown' # Handle 'Unknown' if generated
        }
        for col_name in transformed_df.columns:
            new_name = col_name
            if new_name.startswith('num__'):
                new_name = new_name.replace('num__', '') + '_scaled'
            elif new_name.startswith('cat__'):
                new_name = new_name.replace('cat__', '')
            elif new_name.startswith('remainder__'):
                new_name = new_name.replace('remainder__', '')
            # Apply lowercase mapping if applicable
            new_name = db_column_mapping.get(new_name, new_name)
            column_mapping[col_name] = new_name

        transformed_df.rename(columns=column_mapping, inplace=True)
        logger.info(f"Renamed columns for DB: {list(transformed_df.columns)}")

        # Define expected final columns based on DDL (lowercase regions)
        final_expected_columns = [
            'customer_id', 'age_scaled', 'annual_income_scaled', 'total_transactions',
            'region_aztec', 'region_celtic', 'region_indus', 'region_nomad'
        ]
        # Add region_unknown if it exists and wasn't in the list
        if 'region_unknown' in transformed_df.columns and 'region_unknown' not in final_expected_columns:
             final_expected_columns.append('region_unknown')

        # Select and reorder columns, dropping extras (like potential passthrough index/timestamps)
        logger.info(f"Selecting and reordering final columns: {final_expected_columns}")
        final_columns_present = [col for col in final_expected_columns if col in transformed_df.columns]
        if len(final_columns_present) != len(final_expected_columns):
             missing_cols = list(set(final_expected_columns) - set(final_columns_present))
             logger.warning(f"Expected final columns missing from DataFrame: {missing_cols}. Using only present columns.")

        # Ensure columns are correct type before returning (esp. scaled floats)
        for col in ['age_scaled', 'annual_income_scaled']:
             if col in final_columns_present:
                   transformed_df[col] = pd.to_numeric(transformed_df[col], errors='coerce')
        for col in ['total_transactions','region_aztec','region_celtic','region_indus','region_nomad', 'region_unknown']:
             if col in final_columns_present:
                   transformed_df[col] = pd.to_numeric(transformed_df[col], errors='coerce').astype('Int64')


        transformed_df_final = transformed_df[final_columns_present]

        logger.info(f"Data transformation finished. Final shape: {transformed_df_final.shape}")
        logger.info(f"Data types after transformation:\\n{transformed_df_final.dtypes.to_string()}")
        logger.info(f"First 3 rows of transformed data:\\n{transformed_df_final.head(3).to_string()}")

        return transformed_df_final

    except Exception as e:
        logger.error(f"An error occurred during data transformation: {e}", exc_info=True)
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
    """Loads data from a pandas DataFrame into the specified Supabase table."""
    if df.empty:
        logger.info(f"Transformed DataFrame is empty. No data to load into Supabase table '{table_name}'.")
        return True

    logger.info(f"Attempting to load {len(df)} rows into Supabase table: {table_name}.")
    # Ensure types are suitable for JSON serialization (handle nullable integers -> float/object with None)
    df_upload = df.copy()

    # Explicitly convert OHE region columns to Int64 BEFORE converting to dict
    # This assumes region columns are named like 'region_xxx'
    region_cols = [col for col in df_upload.columns if col.startswith('region_')]
    for col in region_cols:
        try:
            # Convert to numeric first (in case OHE output strings like '1'), then Int64
            df_upload[col] = pd.to_numeric(df_upload[col], errors='coerce').astype('Int64')
            logger.debug(f"Ensured column '{col}' is Int64 before final conversion.")
        except Exception as e_conv_region:
             logger.warning(f"Could not convert region column '{col}' to Int64 before loading: {e_conv_region}")
             # Decide how to handle - drop? fill with default? For now, keep going.

    # Ensure other potential Int64 columns (like total_transactions) are handled
    for col in df_upload.select_dtypes(include=['Int64']).columns:
        # Convert Int64 columns to object type where pd.NA becomes None
        # This prepares for to_dict which handles None correctly for JSON's null
         df_upload[col] = df_upload[col].astype(object).where(pd.notnull(df_upload[col]), None)
         logger.debug(f"Converted Int64 column '{col}' to object with None for pd.NA.")

    # Ensure float columns handle potential NaNs correctly -> None
    for col in df_upload.select_dtypes(include=['float64', 'float32']).columns:
         df_upload[col] = df_upload[col].astype(object).where(pd.notnull(df_upload[col]), None)
         logger.debug(f"Converted float column '{col}' to object with None for NaN.")

    # Now convert to dict - NAs should be None
    records = df_upload.to_dict(orient='records')
    logger.debug(f"Sample record for upload: {records[0] if records else 'N/A'}")


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
    Main function to orchestrate data transformation: Fetch, Transform, Load.
    """
    logger.info("--- Starting transform.py script execution ---")

    supabase = get_supabase_client()
    if supabase is None:
        logger.critical("Supabase client initialization failed in transform.py. Aborting.")
        sys.exit(1)

    cleaned_df = fetch_cleaned_data(supabase, CLEANED_TABLE_NAME)
    if cleaned_df is None:
        logger.critical(f"Failed to fetch cleaned data from '{CLEANED_TABLE_NAME}'. Aborting transformation process.")
        sys.exit(1)

    if cleaned_df.empty:
        logger.info(f"No data fetched from '{CLEANED_TABLE_NAME}'. Nothing to transform or load.")
    else:
        transformed_df = transform_data(cleaned_df)

        if transformed_df is None:
             logger.error("Data transformation failed. Aborting loading process.")
             sys.exit(1)
        elif transformed_df.empty:
             logger.warning("Transformation resulted in an empty DataFrame. No data will be loaded to transformed table.")
        else:
            if REPLACE_EXISTING_TRANSFORMED_DATA:
                logger.info(f"REPLACE_EXISTING_TRANSFORMED_DATA is True. Attempting to delete all data from table '{TRANSFORMED_TABLE_NAME}'.")
                if not delete_all_data_from_table(TRANSFORMED_TABLE_NAME, supabase):
                    logger.warning(f"Failed to delete all data from '{TRANSFORMED_TABLE_NAME}'. Proceeding with insert, but table may contain old data if it existed.")
                else:
                    logger.info(f"Successfully cleared data from table '{TRANSFORMED_TABLE_NAME}'.")

            if not load_data_to_supabase(transformed_df, TRANSFORMED_TABLE_NAME, supabase):
                logger.error(f"Failed to load transformed data into Supabase table '{TRANSFORMED_TABLE_NAME}'.")
            else:
                logger.info(f"Transformed data successfully loaded into Supabase table '{TRANSFORMED_TABLE_NAME}'.")
                # TODO: Update data_history.md

    logger.info("--- transform.py script execution finished ---")

if __name__ == "__main__":
    main()
