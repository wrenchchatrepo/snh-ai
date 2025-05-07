```python
import logging
import os
import sys

# Attempt to import project-level config. If running as a script, this might fail.
try:
    from .. import config as project_config
except ImportError:
    # Fallback if run as a script or project_config is not found at the expected location
    class FallbackConfig:
        LOG_LEVEL = "INFO"
        LOG_FILE = "logs/snh_ai_pipeline.log" # Default log file path
        LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s"

    project_config = FallbackConfig()

# Ensure the logs directory exists
LOG_DIR = os.path.dirname(project_config.LOG_FILE)
if LOG_DIR and not os.path.exists(LOG_DIR):
    try:
        os.makedirs(LOG_DIR)
    except OSError as e:
        # Handle potential race condition if directory is created by another process
        if not os.path.isdir(LOG_DIR):
            sys.stderr.write(f"Error: Could not create log directory {LOG_DIR}. {e}\n")
            # Depending on strictness, you might want to raise an error or exit
            # For now, we'll let it proceed, and file logging might fail.

# Cache for initialized loggers to avoid duplicate handlers
_loggers = {}

def get_logger(name="SNH-AI", level=None, log_file=None, log_format=None):
    """
    Configures and returns a logger instance.

    Args:
        name (str): The name for the logger. Defaults to "SNH-AI".
        level (str/int): Logging level, e.g., "INFO", "DEBUG", logging.INFO.
                         Defaults to project_config.LOG_LEVEL.
        log_file (str): Path to the log file. Defaults to project_config.LOG_FILE.
        log_format (str): Log message format. Defaults to project_config.LOG_FORMAT.

    Returns:
        logging.Logger: Configured logger instance.
    """
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(name)

    # Determine logging level
    log_level_str = level or project_config.LOG_LEVEL
    numeric_level = getattr(logging, log_level_str.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    # Determine log format
    formatter_str = log_format or project_config.LOG_FORMAT
    formatter = logging.Formatter(formatter_str)

    # Console Handler
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        ch = logging.StreamHandler(sys.stdout) # Output to stdout
        ch.setLevel(numeric_level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # File Handler
    log_file_path = log_file or project_config.LOG_FILE
    if log_file_path: # Only add file handler if a path is provided
        # Check if a file handler for this specific file already exists for this logger
        # This is a simple check; more robust would be to check handler.baseFilename
        has_file_handler_for_path = any(
            isinstance(h, logging.FileHandler) and h.baseFilename == os.path.abspath(log_file_path)
            for h in logger.handlers
        )
        if not has_file_handler_for_path:
            try:
                fh = logging.FileHandler(log_file_path, mode='a') # Append mode
                fh.setLevel(numeric_level)
                fh.setFormatter(formatter)
                logger.addHandler(fh)
            except Exception as e:
                logger.error(f"Failed to create file handler for {log_file_path}: {e}", exc_info=True)


    # Prevent messages from propagating to the root logger if handlers are added
    logger.propagate = False

    _loggers[name] = logger
    return logger

if __name__ == "__main__":
    # Example usage when running this script directly
    # This helps in testing the logging setup
    logger = get_logger(__name__, level="DEBUG")

    logger.debug("This is a debug message from logging.py.")
    logger.info("This is an info message from logging.py.")
    logger.warning("This is a warning message from logging.py.")
    logger.error("This is an error message from logging.py.")
    logger.critical("This is a critical message from logging.py.")

    # Test with a different logger name
    module_logger = get_logger("MyModule")
    module_logger.info("Info message from MyModule logger.")

    # Test default logger
    default_logger = get_logger()
    default_logger.info("Info from default SNH-AI logger.")
    print(f"Log file should be at: {os.path.abspath(project_config.LOG_FILE)}")
```