# src/transform.py
# Description: Scales numerical features and encodes categorical features from cleaned data.
# Author: Gemini
# Date: 2024-05-07

import os
import sys
import pandas as pd
import numpy as np
from supabase import create_client, Client
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline # Although ColumnTransformer might be sufficient here

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
    if not config.SUPABASE_URL or not config.SUPABASE_SERVICE_ROLE:
        logger.error("Supabase URL or Service Role Key is not configured.")
        return None
    try:
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
        response = supabase_client.table(table_name).select("*").execute()
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

    # Define the transformers
    # OneHotEncoder: handle_unknown='ignore' will output all zeros for categories not seen during fit
    # sparse_output=False makes it return a dense array, easier to work with pandas
    # drop='first' can avoid multicollinearity if needed, but keep all for now.
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERICAL_COLS),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=int), CATEGORICAL_COLS)
        ],
        remainder='passthrough' # Keep other columns (customer_id, total_transactions, cleaned_at etc)
    )

    try:
        # Fit and transform the data
        transformed_data = preprocessor.fit_transform(df)

        # Get feature names after transformation
        # StandardScaler names remain the same
        # OneHotEncoder names need construction: feature_name + '_' + category_value
        # Passthrough names are fetched using get_feature_names_out
        feature_names_out = preprocessor.get_feature_names_out()
        logger.info(f"Feature names after transformation: {feature_names_out}")

        # Create the transformed DataFrame
        transformed_df = pd.DataFrame(transformed_data, columns=feature_names_out, index=df.index)

        # Rename columns for clarity and DB compatibility (remove prefixes added by ColumnTransformer)
        # e.g., 'num__age' -> 'age_scaled', 'cat__region_Aztec' -> 'region_Aztec', 'remainder__customer_id' -> 'customer_id'
        column_mapping = {}
        for col_name in transformed_df.columns:
            if col_name.startswith('num__'):
                column_mapping[col_name] = col_name.replace('num__', '') + '_scaled'
            elif col_name.startswith('cat__'):
                column_mapping[col_name] = col_name.replace('cat__', '')
            elif col_name.startswith('remainder__'):
                column_mapping[col_name] = col_name.replace('remainder__', '')
            else:
                 column_mapping[col_name] = col_name # Should not happen with remainder='passthrough' default naming

        transformed_df.rename(columns=column_mapping, inplace=True)
        logger.info(f"Renamed columns (initial): {list(transformed_df.columns)}")

        # Fix case sensitivity for database insertion (match actual DB column names)
        db_column_mapping = {
            'region_Aztec': 'region_aztec',
            'region_Celtic': 'region_celtic',
            'region_Indus': 'region_indus',
            'region_Nomad': 'region_nomad'
        }
        # Only rename columns that actually exist in the DataFrame
        rename_dict = {k: v for k, v in db_column_mapping.items() if k in transformed_df.columns}
        if rename_dict:
            transformed_df.rename(columns=rename_dict, inplace=True)
            logger.info(f"Renamed region columns to lowercase for DB compatibility: {list(rename_dict.values())}")
        
        logger.info(f"Columns after case adjustment: {list(transformed_df.columns)}")


        # Reorder columns to a logical order if desired (optional)
        # Define expected final column order based on DDL for transformed_customer_data (lowercase for regions)
        # Note: The order here should match the DDL column order for clarity
        final_expected_columns = [
            'customer_id', 'age_scaled', 'annual_income_scaled', 'total_transactions',
            'region_aztec', 'region_celtic', 'region_indus', 'region_nomad' 
        ]
        # Add region_unknown if it exists (handle_unknown='ignore' might create it with lowercase suffix)
        # OneHotEncoder usually creates 'category_value', so 'region_Unknown' might become 'region_unknown' after the rename logic above if needed.
        # Let's check for 'region_unknown' specifically if that's the expected output from OHE with handle_unknown
        if 'region_unknown' in transformed_df.columns and 'region_unknown' not in final_expected_columns:
             final_expected_columns.append('region_unknown')
        elif 'region_Unknown' in transformed_df.columns and 'region_Unknown' not in final_expected_columns:
              # Handle case where 'region_Unknown' might not have been lowercased if not in db_column_mapping explicitly
              if 'region_unknown' not in transformed_df.columns: # Avoid adding duplicate semantic column
                  transformed_df.rename(columns={'region_Unknown': 'region_unknown'}, inplace=True)
                  final_expected_columns.append('region_unknown')
                  logger.info("Renamed 'region_Unknown' to 'region_unknown' for DB compatibility.")
        
        # Select and reorder columns, dropping any extras (like 'cleaned_at' from passthrough)
        logger.info(f"Selecting and reordering final columns: {final_expected_columns}")
        # Ensure all columns exist before reordering to prevent KeyError
        final_columns_present = [col for col in final_expected_columns if col in transformed_df.columns]
        if len(final_columns_present) != len(final_expected_columns):
             logger.warning(f"Expected final columns {final_expected_columns} but only found {final_columns_present} in DataFrame.")
        
        # Select only the columns intended for the final table
        transformed_df = transformed_df[final_columns_present]

        logger.info(f"Data transformation finished. Final shape: {transformed_df.shape}")
        logger.info(f"Data types after transformation:\\\\n{transformed_df.dtypes.to_string()}")
        logger.info(f"First 3 rows of transformed data:\\\\n{transformed_df.head(3).to_string()}")

        return transformed_df

    except Exception as e:
        logger.error(f"An error occurred during data transformation: {e}", exc_info=True)
        return None


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
        logger.info(f"Transformed DataFrame is empty. No data to load into Supabase table '{table_name}'.")
        return True

    logger.info(f"Attempting to load {len(df)} rows into Supabase table: {table_name}.")
    # Convert potentially nullable Int64 back to float/object for JSON if needed, handle NaN->None
    # StandardScaler outputs float64, OneHotEncoder outputs int specified by dtype=int
    records = df.astype(object).where(pd.notnull(df), None).to_dict(orient='records')

    try:
        response = supabase_client.table(table_name).insert(records).execute()
        if hasattr(response, 'error') and response.error:
            logger.error(f"Error inserting data into Supabase table '{table_name}': {response.error.message}")
            # Log more details if available
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
    Main function to orchestrate data transformation:
    1. Initialize Supabase client.
    2. Fetch cleaned data from Supabase.
    3. Transform the data (scale & encode).
    4. Load transformed data into a new Supabase table.
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
        transformed_df = transform_data(cleaned_df) # Pass the actual dataframe

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
                    # Or sys.exit(1) if strict clear is required
                else:
                    logger.info(f"Successfully cleared data from table '{TRANSFORMED_TABLE_NAME}'.")

            if not load_data_to_supabase(transformed_df, TRANSFORMED_TABLE_NAME, supabase):
                logger.error(f"Failed to load transformed data into Supabase table '{TRANSFORMED_TABLE_NAME}'.")
            else:
                logger.info(f"Transformed data successfully loaded into Supabase table '{TRANSFORMED_TABLE_NAME}'.")
                # TODO: Update data_history.md with info about transformed_customer_data table & record count.

    logger.info("--- transform.py script execution finished ---")

if __name__ == "__main__":
    main()