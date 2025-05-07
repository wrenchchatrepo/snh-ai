# process.py
# Description: Orchestrates the execution of the SNH-AI data pipeline scripts.
# Author: Gemini
# Date: 2024-05-07

import sys
import os
import traceback
import time

# Ensure the src directory is in the path for module imports
# Assuming process.py is run from the project root (dev/snh-ai)
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# Import main functions from pipeline scripts and logger
try:
    from src import snh_logger as snh_logging
    from src.Ingest import main as ingest_main
    from src.clean import main as clean_main
    from src.transform import main as transform_main
    from src.ml_model import main as ml_model_main
    # Placeholder for prediction model if needed
    # from src.predictive_model import main as predictive_model_main
except ImportError as e:
    # Use standard logging if custom logger fails initially
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.critical(f"Failed to import necessary pipeline modules or logger: {e}. Aborting.")
    sys.exit(1)

# Get logger instance
logger = snh_logging.get_logger("PipelineOrchestrator")

def run_step(step_name: str, step_function):
    """Runs a pipeline step, logs status, and handles errors."""
    logger.info(f"--- Starting Step: {step_name} ---")
    start_time = time.time()
    try:
        step_function() # Call the main function of the step
        end_time = time.time()
        logger.info(f"--- Finished Step: {step_name} (Duration: {end_time - start_time:.2f} seconds) ---")
        return True
    except SystemExit as e:
        # Allow SystemExit to propagate if a script intentionally exits
        logger.error(f"--- Step {step_name} exited with code {e.code}. ---")
        raise # Re-raise to stop the pipeline
    except Exception as e:
        logger.error(f"--- Step Failed: {step_name} ---")
        logger.error(f"Error: {e}")
        logger.error(traceback.format_exc()) # Log full traceback
        return False

def run_pipeline():
    """Executes the full SNH-AI data pipeline."""
    logger.info("==============================================")
    logger.info("Starting SNH-AI Full Data Pipeline Execution")
    logger.info("==============================================")
    pipeline_start_time = time.time()

    steps = [
        ("Data Ingestion", ingest_main),
        ("Data Cleaning", clean_main),
        ("Data Transformation", transform_main),
        ("ML Model Training & Segmentation", ml_model_main),
        # ("Predictive Modeling", predictive_model_main), # Uncomment if needed
    ]

    all_steps_successful = True
    for name, func in steps:
        success = run_step(name, func)
        if not success:
            logger.critical(f"Pipeline halted due to failure in step: {name}")
            all_steps_successful = False
            break # Stop pipeline on first failure

    pipeline_end_time = time.time()
    total_duration = pipeline_end_time - pipeline_start_time
    logger.info("==============================================")
    if all_steps_successful:
        logger.info(f"SNH-AI Full Data Pipeline Execution COMPLETED Successfully")
    else:
        logger.error(f"SNH-AI Full Data Pipeline Execution FAILED")
    logger.info(f"Total Pipeline Duration: {total_duration:.2f} seconds")
    logger.info("==============================================")

    if not all_steps_successful:
        sys.exit(1) # Exit with error code if pipeline failed

if __name__ == "__main__":
    run_pipeline()