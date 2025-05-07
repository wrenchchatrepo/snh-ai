import logging
import os
import sys

# Adjust path to import config from parent directory of src/
# This assumes config.py is in dev/snh-ai/ and snh_logger.py is in dev/snh-ai/src/
PROJECT_ROOT_FOR_CONFIG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT_FOR_CONFIG not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_FOR_CONFIG)

try:
    import config as project_config
except ImportError as e_config:
    sys.stderr.write(f"CRITICAL: Failed to import 'config' from {PROJECT_ROOT_FOR_CONFIG}. Error: {e_config}\nEnsure config.py exists in the project root and PYTHONPATH is correct.\n")
    class FallbackConfig: # Minimal fallback if config import fails
        LOG_LEVEL = "INFO"
        LOG_FILE = None # No file logging by default if config fails
        LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s"
    project_config = FallbackConfig()
finally:
    # Clean up sys.path if we modified it, to avoid side effects
    if sys.path and sys.path[0] == PROJECT_ROOT_FOR_CONFIG: # Check if the path is still the one we added
        sys.path.pop(0)


# Ensure the logs directory exists if a file path is specified and logging to file is intended
if hasattr(project_config, 'LOG_FILE') and project_config.LOG_FILE:
    LOG_DIR = os.path.dirname(project_config.LOG_FILE)
    if LOG_DIR and not os.path.exists(LOG_DIR): # Check if LOG_DIR is not an empty string
        try:
            os.makedirs(LOG_DIR)
        except OSError as e:
            if not os.path.isdir(LOG_DIR): 
                sys.stderr.write(f"Error: Could not create log directory {LOG_DIR}. {e}\n")

_loggers = {}

def get_logger(name="SNH-AI", level=None, log_file_override=None, log_format_override=None):
    if name in _loggers and not (level or log_file_override or log_format_override):
        return _loggers[name]

    logger = logging.getLogger(name)
    
    current_level_str = level or getattr(project_config, 'LOG_LEVEL', "INFO")
    numeric_level = getattr(logging, current_level_str.upper(), logging.INFO)
    
    # Clear handlers only if reconfiguring with new parameters or if it's a truly new logger instance being configured.
    if logger.handlers and (level or log_file_override or log_format_override):
        logger.handlers = [] 
    
    logger.setLevel(numeric_level) # Always set/reset level

    current_format_str = log_format_override or getattr(project_config, 'LOG_FORMAT', "%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s")
    formatter = logging.Formatter(current_format_str)

    # Add console handler if not already present or if reconfiguring
    if not any(isinstance(h, logging.StreamHandler) and h.stream == sys.stdout for h in logger.handlers):
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(numeric_level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # Add file handler if not already present for this path or if reconfiguring
    effective_log_file = log_file_override or getattr(project_config, 'LOG_FILE', None)
    if effective_log_file:
        if not any(isinstance(h, logging.FileHandler) and h.baseFilename == os.path.abspath(effective_log_file) for h in logger.handlers):
            log_dir_for_file = os.path.dirname(effective_log_file)
            if log_dir_for_file and not os.path.exists(log_dir_for_file):
                try:
                    os.makedirs(log_dir_for_file)
                except OSError as e:
                    if not os.path.isdir(log_dir_for_file):
                         sys.stderr.write(f"Error: Could not create log directory {log_dir_for_file} for FileHandler. {e}\n")
            try:
                fh = logging.FileHandler(effective_log_file, mode='a')
                fh.setLevel(numeric_level)
                fh.setFormatter(formatter)
                logger.addHandler(fh)
            except Exception as e:
                sys.stderr.write(f"Error creating file handler for {effective_log_file}: {e}\n")
    
    logger.propagate = False
    
    _loggers[name] = logger
    return logger

if __name__ == "__main__":
    _loggers.clear() 
    
    print("Testing snh_logger.py...")

    print("\n--- Testing with default/config.py settings ---")
    logger_default_test = get_logger("DefaultTest")
    logger_default_test.debug("This is a DEBUG message (DefaultTest).")
    logger_default_test.info("This is an INFO message (DefaultTest).")
    logger_default_test.warning("This is a WARNING message (DefaultTest).")

    print("\n--- Testing with overridden level (ERROR) ---")
    logger_level_override = get_logger("LevelOverrideTest", level="ERROR")
    logger_level_override.info("This INFO message should NOT appear (LevelOverrideTest).")
    logger_level_override.error("This is an ERROR message (LevelOverrideTest).")

    temp_log_file = os.path.join(os.path.dirname(__file__), "temp_logging_test.log")
    if os.path.exists(temp_log_file):
        os.remove(temp_log_file) 

    print(f"\n--- Testing with overridden log file: {temp_log_file} ---")
    logger_file_override = get_logger("FileOverrideTest", log_file_override=temp_log_file, level="DEBUG")
    logger_file_override.debug(f"This DEBUG message should go to console AND {temp_log_file} (FileOverrideTest).")
    logger_file_override.info(f"This INFO message should go to console AND {temp_log_file} (FileOverrideTest).")
    
    if os.path.exists(temp_log_file):
        print(f"Content of {temp_log_file}:")
        with open(temp_log_file, 'r') as f:
            print(f.read())
        os.remove(temp_log_file) 
    else:
        print(f"ERROR: {temp_log_file} was not created by the file override test.")

    print("\n--- Testing main SNH-AI logger (as used by other modules) ---")
    snh_ai_logger = get_logger() 
    snh_ai_logger.info("Info from default SNH-AI logger, fetched via get_logger().")
    
    effective_log_file_for_main_check = getattr(project_config, 'LOG_FILE', None)
    if effective_log_file_for_main_check:
        print(f"Default log file (if configured in config.py and writable) should be at: {os.path.abspath(effective_log_file_for_main_check)}")
    else:
        print("Default file logging (project_config.LOG_FILE) is not configured or config.py was not loaded.")
    
    print("\nsnh_logger.py test finished.")