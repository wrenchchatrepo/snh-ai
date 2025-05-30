## Data History and Lineage

This document tracks the history of database tables created and modified by the SNH-AI project, along with the relevant SQL or Python script context. Without the use the mcp-supabase I had to manually create the table schemas withe DDl statements you see below. Normally, I would use the mcp server to automate these tasks or, given more time I would have written scripts to leverage the Supabase CLI. To circumvent this, I including the Postgres tables in the Dockerfile.

| Date | Table Name | Operation | Script/Context | Description/Notes |
| :--- | :--------- | :-------- | :------------- | :---------------- |
| 2025-05-07 | `raw_customer_data` | Schema: Table Creation | `Manual DDL via Supabase SQL Editor` | Initial schema for raw customer data. DDL: `CREATE TABLE IF NOT EXISTS raw_customer_data (customer_id TEXT PRIMARY KEY, age INTEGER, annual_income NUMERIC, total_transactions INTEGER, region TEXT, ingested_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP);` |
| 2025-05-07 | `raw_customer_data` | Data Load (Replace) | `src/Ingest.py` | Loaded 60 records from source CSV. Existing data was cleared before load.|
| 2025-05-07 | `cleaned_customer_data` | Schema: Table Creation | `Manual DDL via Supabase SQL Editor` | Schema for cleaned data. DDL: `CREATE TABLE IF NOT EXISTS cleaned_customer_data (customer_id TEXT PRIMARY KEY, age INTEGER, annual_income NUMERIC, total_transactions INTEGER, region TEXT, cleaned_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP);` |
| 2025-05-07 | `cleaned_customer_data` | Data Load (Replace) | `src/clean.py` | Loaded 60 cleaned records from raw_customer_data. Existing data cleared before load. Missing values handled, no duplicates removed. |
| 2025-05-07 | `transformed_customer_data` | Schema: Table Creation | `Manual DDL via Supabase SQL Editor` | Schema for transformed data. DDL: `CREATE TABLE IF NOT EXISTS transformed_customer_data (customer_id TEXT PRIMARY KEY, age_scaled DOUBLE PRECISION, annual_income_scaled DOUBLE PRECISION, total_transactions INTEGER, region_aztec INTEGER, region_celtic INTEGER, region_indus INTEGER, region_nomad INTEGER, transformed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP);` |
| 2025-05-07 | `transformed_customer_data` | Data Load (Replace) | `src/transform.py` | Loaded 60 transformed records from cleaned_customer_data. Existing data cleared before load. |
| 2025-05-07 | `customer_segments` | Schema: Table Creation | `Manual DDL via Supabase SQL Editor` | Schema for final customer segments. DDL: `CREATE TABLE IF NOT EXISTS customer_segments (customer_id TEXT PRIMARY KEY, pattern_id INTEGER NOT NULL, assigned_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP);` |
| 2025-05-07 | `customer_segments` | Data Load (Replace) | `src/ml_model.py` | Loaded 60 customer segment assignments (pattern_id) using KMeans (k=6). Existing data cleared before load. |
| 2025-05-07 | `transaction_predictions` | Schema: Table Creation | `Manual DDL via Supabase SQL Editor` | Schema for transaction predictions. DDL: `CREATE TABLE IF NOT EXISTS transaction_predictions (customer_id TEXT PRIMARY KEY, predicted_total_transactions REAL, predicted_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP);` |
| 2025-05-07 | `transaction_predictions` | Data Load (Replace) | `src/predictive_model.py` | Loaded 60 transaction predictions using RandomForestRegressor. Existing data cleared before load. |

**Notes:**
+ Operation: e.g., Table Creation, Schema Update, Data Update, Data Deletion.
+ Script/Context: Reference the Python script name (e.g., `src/Ingest.py`), a specific SQL file, or a function name responsible for the operation.
+ Description/Notes: Brief explanation of the change or the purpose of the table at this stage.