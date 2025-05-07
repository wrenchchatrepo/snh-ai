# Data History and Lineage

This document tracks the history of database tables created and modified by the SNH-AI project, along with the relevant SQL or Python script context.

| Date       | Table Name          | Operation      | Script/Context                                      | Description/Notes                                   |
| :--------- | :------------------ | :------------- | :-------------------------------------------------- | :-------------------------------------------------- |
| YYYY-MM-DD | `raw_customer_data` | Table Creation | `src/Ingest.py` (initial load from CSV or DB source)  | Stores the initial, unprocessed customer data.      |
| YYYY-MM-DD | `cleaned_customer_data` | Table Creation | `src/clean.py`                                    | Output of data cleaning (missing values, duplicates). |
| YYYY-MM-DD | `transformed_customer_data` | Table Creation | `src/scale.py`, `src/encode.py`                     | Output after scaling and one-hot encoding.           |
| YYYY-MM-DD | `customer_segments` | Table Creation | `src/ml_model.py`, `src/presentation.py`            | Final data with customer segment `pattern_id`.        |
|            |                     |                |                                                     |                                                     |
|            |                     |                |                                                     |                                                     |

**Notes:**
*   **Operation**: e.g., Table Creation, Schema Update, Data Update, Data Deletion.
*   **Script/Context**: Reference the Python script name (e.g., `src/Ingest.py`), a specific SQL file, or a function name responsible for the operation.
*   **Description/Notes**: Brief explanation of the change or the purpose of the table at this stage.