# SNH-AI Exercise
The is original outline to this exercise. The outline to the solution is below and is relatively true to what I delivered. 
## Objective: 
Design and implement an end-to-end data pipeline that extracts data from a database, transforms it, and applies a machine learning model that identifies patterns in the transformed data.
## Specifics:
### 1. Data Extraction:
○ Assume we have a database (e.g., PostgreSQL, MySQL, or even a CSV file for
simplicity).
○ The database contains customer data with the following fields: customer_id, age,
annual_income, total_transactions, and region.
○ Write a script (in Python) to extract this data. The script should handle potential
connection errors and log any issues.
### 2. Data Transformation:
○ Clean the data:
  ■ Handle missing values (e.g., by replacing them with the median).
  ■ Remove any duplicate records.
○ Scale the age and annual_income columns using StandardScaler.
○ One-hot encode the region column.
○ The transformation logic should be modular and reusable.
### 3. Machine Learning:
○ Apply ML model to the transformed data to identify customer segments.
○ Use scikit-learn for the ML Models.
○ Determine the optimal number of patterns using the elbow method.
○ Add the unique pattern labels as a new column (pattern_id) to the transformed
data.
### 4. Data Loading:
○ Load the data with the labels into a new table in the database (or a new CSV file).
○ The loading process should also handle potential errors.
### 5. Deliverables:
○ A Python script that performs the entire ETL and ML process.
○ A README file with instructions on how to set up the environment, connect to the
database, and run the script. Include any dependencies.
○ (Optional) A brief report (1-2 pages) describing the data transformation steps,
the chosen ML Model, and any insights gained from the ML results on the data.
## Solution
### Write python scripts
#### Open Zed
+ Script: setup.py
    + Create local repo: dev/snh-ai
    + Initialize: 
        + .env via secrets-manager
            + ANTHROPIC_API_KEY (Account:snh-ai)
            + GITHUB_PAT (Account:snh-ai)
            + SUPABASE_* (Account:postgres, 4 Keys)
        + mcp-servers/
            + bayes-mcp: https://github.com/wrenchchatrepo/bayes-msp
            + mcp-server-context7: https://github.com/akbxr/zed-mcp-server-context7
                + https://context7.com/scikit-rf/scikit-rf
                + https://context7.com/supabase/supabase
                + https://context7.com/python/cpython
                + https://context7.com/axiomhq/docs
                + https://context7.com/pandas-dev/pandas
                + https://context7.com/jupyter/nbclient
            + postgres-context-server: https://github.com/zed-extensions/postgres-context-server
            + mcp-server-axiom: https://github.com/zed-extensions/mcp-server-axiom
            + mcp-server-github: https://github.com/LoamStudios/zed-mcp-server-github
            + mcp-server-sequential-thinking: https://github.com/LoamStudios/zed-mcp-server-sequential-thinking
        + src/
            + template.json
                + try/except
                + logging axiom.py
            + logging.py
        + docs/
            + StandardScaler
            + One-hot encode
        + config.py
        + readme.md
        + .gitignore
        + .github
    + Add, commit, push, PR
#### 1. Data Extraction:
+ Script: Ingest.py
    + Postgres via Supabase MCP
    + Raw_Data: /Users/dionedge/dev/data_engineer_customer_data.csv
        + Headers:
            + customer_id
            + age
            + annual_income
            + total_transactions
            + region
        + logging.py
        + Verify
+ Update config.py
#### 2. Data Transformation:
+ context7: https://context7.com/scikit-rf/scikit-rf
+ Script: clean.py
    + import pandas as pd
    + Handling Missing Values:
        + Identifying missing values using df.isnull() or df.isna()
        + Filling missing values with df.fillna(), using median
    + Removing Duplicates:
        + Identifying duplicate rows using df.duplicated()
        + Removing duplicate rows using df.drop_duplicates()
    + logging.py
+ Script: scale.py
    + from sklearn.preprocessing import StandardScaler
    + logging.py
+ Script: encode.py
    + from sklearn.preprocessing import OneHotEncoder
    + logging.py
#### 3. Machine Learning:
+ context7: https://context7.com/scikit-rf/scikit-rf
+ Script:
    + from sklearn.cluster import KMeans
    + from sklearn.ensemble import RandomForestClassifier
    + bayes-mcp
    + Unique pattern labels as new column (pattern_id)
    + logging.py
#### 4. Data Loading:
+ Script: presentation.py
    + Loal new tables with Transformed data
    + pip install jupyterlab
    + Create notebook
    + logging.py
#### 5. Deliverables:
+ Script: setup.py
    + .env file
    + start-database.sh
    + logging.py
+ Script: process.py
    + all scripts
    + logging 
+ Update README
+ Create Dockerfile
### Apps by Dion Edge
+ secrets-manager
+ bayes-mcp