# Data History and Lineage

This document tracks the history of database tables created and modified by the SNH-AI project, along with the relevant SQL or Python script context.

| Date       | Table Name          | Operation      | Script/Context                                      | Description/Notes                                   |
| :--------- | :------------------ | :------------- | :-------------------------------------------------- | :-------------------------------------------------- |
| 2025-05-07 | `raw_customer_data` | Schema: Table Creation | `Manual DDL via Supabase SQL Editor`                 | Initial schema for raw customer data. Columns: customer_id (PK), age, annual_income, total_transactions, region, ingested_at. |
| 2025-05-07 | `raw_customer_data` | Data Load (Replace)    | `src/Ingest.py`                                      | Loaded 60 records from source CSV. Existing data was cleared before load.                                               |
| 2025-05-07 | `cleaned_customer_data` | Schema: Table Creation | `Manual DDL via Supabase SQL Editor`                 | Schema for cleaned data. Columns match raw except 'ingested_at' dropped, 'cleaned_at' added.                             |
| 2025-05-07 | `cleaned_customer_data` | Data Load (Replace)    | `src/clean.py`                                     | Loaded 60 cleaned records from raw_customer_data. Existing data cleared before load. Missing values handled, no duplicates removed. |
| 2025-05-07 | `transformed_customer_data` | Schema: Table Creation | `Manual DDL via Supabase SQL Editor`                 | Schema for transformed data (scaled numerics, OHE categoricals). Columns: customer_id (PK), age_scaled, annual_income_scaled, total_transactions, region_aztec, region_celtic, region_indus, region_nomad, transformed_at. |
| 2025-05-07 | `transformed_customer_data` | Data Load (Replace)    | `src/transform.py`                                     | Loaded 60 transformed records from cleaned_customer_data. Existing data cleared before load.                           |
| 2025-05-07 | `customer_segments`     | Schema: Table Creation | `Manual DDL via Supabase SQL Editor`                 | Schema for final customer segments. Columns: customer_id (PK), pattern_id, assigned_at.                                |
| 2025-05-07 | `customer_segments`     | Data Load (Replace)    | `src/ml_model.py`                                    | Loaded 60 customer segment assignments (pattern_id) using KMeans (k=6). Existing data cleared before load.                |
|            |                         |                        |                                                      |                                                                                                                         |

**Notes:**
*   **Operation**: e.g., Table Creation, Schema Update, Data Update, Data Deletion.
*   **Script/Context**: Reference the Python script name (e.g., `src/Ingest.py`), a specific SQL file, or a function name responsible for the operation.
*   **Description/Notes**: Brief explanation of the change or the purpose of the table at this stage.