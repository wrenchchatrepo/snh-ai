# config.py
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Database Configuration ---
# Example for PostgreSQL (to be filled with Supabase details)
# DB_USER = "your_db_user"
# DB_PASSWORD = "your_db_password"
# DB_HOST = "your_db_host"
# DB_PORT = "your_db_port"
# DB_NAME = "your_db_name"

# Supabase specific
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON") # Public key for client-side access (loaded from SUPABASE_ANON env var)
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE") # Secret key for server-side/admin access (loaded from SUPABASE_SERVICE_ROLE env var)
SUPABASE_DB_PASSWORD = os.getenv("SUPABASE_DB_PASSWORD") # For direct DB connections if needed, or used by Supabase client
SUPABASE_PROJECT_ID = os.getenv("SUPABASE_PROJECT_ID") # Project ID from Supabase settings

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
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s"

# --- Axiom Logging Configuration (if used) ---
AXIOM_TOKEN = os.getenv("AXIOM_TOKEN") # Changed from AXIOM_API_KEY based on user feedback
AXIOM_DATASET_NAME = os.getenv("AXIOM_DATASET_NAME") # e.g., "snh-ai-pipeline"

# --- ML Model Configuration ---
# For KMeans clustering
MAX_CLUSTERS_FOR_ELBOW = 10  # Max number of clusters to test for the elbow method

# --- API Keys (loaded from .env, not hardcoded here) ---
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GITHUB_PAT = os.getenv("GITHUB_PAT")
# Supabase keys are loaded under Database Configuration

# Optional: Supabase JWT Secret if needed for custom token generation/verification
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET")

# --- Other Configurations ---
# Example: RANDOM_STATE for reproducibility in ML models
RANDOM_STATE = 42