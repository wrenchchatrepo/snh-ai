# dev/snh-ai/test_config.py
import os
import sys

# Print CWD for context
print(f"Current working directory for test_config.py: {os.getcwd()}")

# Check for .env in the current directory (expected to be dev/snh-ai/)
DOTENV_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '.env'))
print(f"Checking for .env file at: {DOTENV_PATH}")
if not os.path.exists(DOTENV_PATH):
    print(f"ERROR: .env file does NOT exist at {DOTENV_PATH}")
else:
    print(f".env file FOUND at {DOTENV_PATH}")
    try:
        # Try to read .env content for basic verification (don't print values)
        with open(DOTENV_PATH, 'r') as f:
            env_content_snippet = f.read(100) # Read first 100 chars
        print(f".env content starts with: \"{env_content_snippet.strip()}...\" (values not shown for security)")
        print(f".env file size: {os.path.getsize(DOTENV_PATH)} bytes")
    except Exception as e_read_env:
        print(f"Could not read .env file content snippet: {e_read_env}")


# Dynamically adjust sys.path to ensure config.py can be imported if it's in the same directory
# This script (test_config.py) is intended to be in the same directory as config.py
# so direct import should work, but let's be robust.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

print(f"sys.path temporarily includes: {SCRIPT_DIR} (for importing local config.py)")


try:
    print("\\n--- Attempting to import and inspect config.py ---")
    import config # This will execute config.py, including its load_dotenv()
    print("--- Successfully imported config.py ---")
    
    # Check all relevant keys from config.py
    keys_to_check = [
        'RAW_DATA_CSV', 'SUPABASE_URL', 'SUPABASE_SERVICE_ROLE_KEY', 'SUPABASE_ANON_KEY', 'SUPABASE_DB_PASSWORD',
        'AXIOM_TOKEN', 'AXIOM_DATASET_NAME', 
        'ANTHROPIC_API_KEY', 'GITHUB_PAT', 'SUPABASE_JWT_SECRET',
        'OUTPUT_DB_TABLE_NAME', 'LOG_LEVEL', 'LOG_FILE', 'LOG_FORMAT',
        'MAX_CLUSTERS_FOR_ELBOW', 'RANDOM_STATE'
    ]

    for key in keys_to_check:
        value = getattr(config, key, '!!! NOT FOUND IN CONFIG !!!')
        value_type = type(value).__name__
        # For sensitive keys, don't print the value if it's a string and not 'None' or 'Not Found'
        if isinstance(value, str) and key in ['SUPABASE_SERVICE_ROLE_KEY', 'SUPABASE_DB_PASSWORD', 'AXIOM_TOKEN', 'ANTHROPIC_API_KEY', 'GITHUB_PAT', 'SUPABASE_JWT_SECRET', 'SUPABASE_ANON_KEY']:
            display_value = f"'******' if value and value != '!!! NOT FOUND IN CONFIG !!!' else value"
        else:
            display_value = f"'{value}'"
        
        print(f"config.{key}: {display_value} (Type: {value_type})")
        if value is None:
            print(f"  WARNING: config.{key} is None. Check .env file and config.py os.getenv('{key}').")
        elif value == '!!! NOT FOUND IN CONFIG !!!':
             print(f"  ERROR: config.{key} is not defined in config.py at all.")


except ImportError as e_import:
    print(f"ERROR: Failed to import config.py. Details: {e_import}")
    print(f"Make sure config.py exists in the same directory as test_config.py ({SCRIPT_DIR}) and has no syntax errors.")
except AttributeError as e_attr:
    print(f"ERROR: Error accessing an attribute in config.py, likely after import. Details: {e_attr}")
except Exception as e_general:
    print(f"An UNEXPECTED ERROR occurred: {e_general}")

finally:
    # Clean up sys.path if we modified it
    if SCRIPT_DIR in sys.path and sys.path[0] == SCRIPT_DIR:
        sys.path.pop(0)
    print("\\nTest finished.")