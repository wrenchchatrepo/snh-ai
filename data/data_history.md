# Data History and Lineage

This document tracks the history of database tables created and modified by the SNH-AI project, along with the relevant SQL or Python script context.

| Date       | Table Name          | Operation      | Script/Context                                      | Description/Notes                                   |
| :--------- | :------------------ | :------------- | :-------------------------------------------------- | :-------------------------------------------------- |
| 2025-05-07 | `raw_customer_data` | Schema: Table Creation | `Manual DDL via Supabase SQL Editor`                 | Initial schema for raw customer data. Columns: customer_id (PK), age, annual_income, total_transactions, region, ingested_at. |
| 2025-05-07 | `raw_customer_data` | Data Load (Replace)    | `src/Ingest.py`                                      | Loaded 60 records from source CSV. Existing data was cleared before load.                                               |
| YYYY-MM-DD | `cleaned_customer_data` | Table Creation         | `src/clean.py`                                     | Output of data cleaning (missing values, duplicates). Expected record count: TBD.                                        |
| YYYY-MM-DD | `transformed_customer_data` | Table Creation     | `src/scale.py`, `src/encode.py`                      | Output after scaling and one-hot encoding. Expected record count: TBD.                                                 |
| YYYY-MM-DD | `customer_segments`     | Table Creation         | `src/ml_model.py`, `src/presentation.py`             | Final data with customer segment `pattern_id`. Expected record count: TBD.                                             |
|            |                         |                        |                                                      |                                                                                                                         |

**Notes:**
*   **Operation**: e.g., Table Creation, Schema Update, Data Update, Data Deletion.
*   **Script/Context**: Reference the Python script name (e.g., `src/Ingest.py`), a specific SQL file, or a function name responsible for the operation.
*   **Description/Notes**: Brief explanation of the change or the purpose of the table at this stage.