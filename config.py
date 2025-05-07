# config.py

# --- Database Configuration ---
# Example for PostgreSQL (to be filled with Supabase details)
# DB_USER = "your_db_user"
# DB_PASSWORD = "your_db_password"
# DB_HOST = "your_db_host"
# DB_PORT = "your_db_port"
# DB_NAME = "your_db_name"

# Supabase specific (if using Supabase client library)
# SUPABASE_URL = "your_supabase_url"
# SUPABASE_KEY = "your_supabase_key"

# --- Data Paths ---
# Path to the raw data CSV file
RAW_DATA_CSV = "/Users/dionedge/dev/data_engineer_customer_data.csv"

# Paths for intermediate data (if saving to disk)
# PROCESSED_DATA_DIR = "data/processed/"
# TRANSFORMED_DATA_PATH = PROCESSED_DATA_DIR + "transformed_data.csv"
# FINAL_DATA_WITH_PATTERNS_PATH = PROCESSED_DATA_DIR + "final_data_with_patterns.csv"

# Output table name in the database
OUTPUT_DB_TABLE_NAME = "customer_segments"


# --- Logging Configuration ---
LOG_LEVEL = "INFO"  # e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FILE = "logs/snh_ai_pipeline.log"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# --- ML Model Configuration ---
# For KMeans clustering
MAX_CLUSTERS_FOR_ELBOW = 10  # Max number of clusters to test for the elbow method

# --- API Keys (loaded from .env, not hardcoded here) ---
# ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
# GITHUB_PAT = os.getenv("GITHUB_PAT")
# SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
# SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
# SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET")
# SUPABASE_DB_PASSWORD = os.getenv("SUPABASE_DB_PASSWORD")


# --- Other Configurations ---
# Example: RANDOM_STATE for reproducibility in ML models
RANDOM_STATE = 42