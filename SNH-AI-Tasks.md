# SNH-AI Project Tasks
This is the AI powered task like that prompt for with my own specification (Original Problem Statement with Outline of Solution.md) that I keep updated throughout develeopment to manage my work and time.

## I. Project Setup & Configuration (`setup.py` driven)

### A. Repository and Version Control
1.  **Create Local Repository**:
    [X] Initialize a Git repository in `dev/snh-ai`.
2.  **Initial Git Workflow**:
    [X] Perform an initial commit with the basic project structure.
    [X] Set up a remote repository on GitHub.
    [X] Push initial changes and create a Pull Request as per project guidelines.

### B. Environment and Secrets
3.  **Develop `setup.py` (Initial)**:
    [X] Create a Python script `setup.py` to automate project initialization.
4.  **Initialize `.env` File**:
    [X] Task for `setup.py`: Create `.env` file.
    [X] Source secrets from `secrets-manager` for:
        [X] `ANTHROPIC_API_KEY`
        [X] `GITHUB_PAT`
        [X] `SUPABASE_*` (4 keys)
5.  **Create `.gitignore` File**:
    [X] Task for `setup.py` or manual: Add common Python, environment, and OS-specific files to ignore (e.g., `__pycache__/`, `*.pyc`, `.env`, `venv/`).
6.  **Create `.github` Directory**:
    [X] Task for `setup.py` or manual: Set up the `.github/` directory for workflows or issue templates if needed.

### C. Directory Structure
7.  **Create Core Directories**:
    [X] Task for `setup.py`:
        [X] `mcp-servers/`
        [X] `src/`
        [X] `docs/`
8.  **Set up MCP Servers in `mcp-servers/`**:
    [X] Task for `setup.py` or manual provisioning:
        [X]`bayes-mcp`
        [X]`mcp-server-context7` (configure with specified Context7 URLs)
        [X]`postgres-context-server`
        [X]`mcp-server-axiom`
        [X]`mcp-server-github`
        [X]`mcp-server-sequential-thinking`

### D. Core Scripts and Configuration
9.  **Develop `src/logging.py`**:
    [X] Create a reusable logging module.
    [X] Consider integration with Axiom (via `mcp-server-axiom`).
10. **Create `src/template.json`**:
    [X] Define a template for script structure, including try/except blocks and calls to `logging.py`.
11. **Develop `config.py`**:
    [X] Create a central configuration file in the project root for parameters like database connections, file paths, etc.
12. **Create Initial `README.md`**:
    [X] Task for `setup.py` or manual: Create a basic `README.md` in the project root. This will be updated later.
13. **Populate `docs/` Directory**:
    [X] Add documentation files for:
        [X]`StandardScaler` usage and context.
        [X]`One-hot encode` usage and context.

## II. Data Pipeline Implementation (Python Scripts in `src/`)

### A. Data Extraction
14. **Develop `src/Ingest.py`**:
    [X] Connect to PostgreSQL via Supabase MCP.
    [X] Extract data from the specified CSV (`/Users/dionedge/dev/data_engineer_customer_data.csv`).
    [X] Define schema: `customer_id`, `age`, `annual_income`, `total_transactions`, `region`.
    [X] Implement error handling for connection issues.
    [X] Integrate logging using `src/logging.py`.
    [X] Implement data verification steps post-extraction.
15. **Update `config.py` for Ingestion**:
    [X] Add configurations for data source (CSV path, DB connection details if switching from CSV).

### B. Data Transformation
16. **Develop `src/clean.py`**:
    [X] Import `pandas`.
    [X] Load data from the output of `Ingest.py`.
    [X] Handle missing values (e.g., fill with median for numerical columns).
    [X] Remove duplicate records.
    [X] Integrate logging.
17. **Develop `src/scale.py`**:
    [X] Import `StandardScaler` from `sklearn.preprocessing`.
    [X] Load data from the output of `clean.py`.
    [X] Apply `StandardScaler` to `age` and `annual_income` columns.
    [X] Integrate logging.
18. **Develop `src/encode.py`**:
    [X] Import `OneHotEncoder` from `sklearn.preprocessing`.
    [X] Load data from the output of `scale.py`.
    [X] Apply `OneHotEncoder` to the `region` column.
    [X] Integrate logging.

### C. Machine Learning
19. **Develop `src/ml_model.py` (or similar name)**:
    [X] Import `KMeans` from `sklearn.cluster`.
    [X] (Optional) Import `RandomForestClassifier` from `sklearn.ensemble` if its use case is clarified.
    [X] Load transformed data (output of `encode.py`).
    [X] Implement the elbow method to determine the optimal number of clusters (patterns) for KMeans.
    [X] Train the KMeans model.
    [X] Add a new column `pattern_id` with the cluster labels to the dataset.
    [X] Integrate with `bayes-mcp` if applicable (e.g., for hyperparameter tuning or other Bayesian methods).
    [X] Integrate logging.

### D. Data Loading
20. **Develop `src/presentation.py`**:
    [X] Load the final transformed data with `pattern_id` (output of `ml_model.py`).
    [X] Load this data into a new table in the PostgreSQL database (via Supabase MCP) or a new CSV file.
    [X] Implement error handling for loading operations.
    [X] Integrate logging.

## III. Orchestration & Deliverables

21. **Develop `process.py` (Orchestration Script)**:
    [X] Create a main script (e.g., in project root or `src/`) to run the entire pipeline.
    [X] Sequence of calls: `Ingest.py` -> `clean.py` -> `scale.py` -> `encode.py` -> `ml_model.py` -> `presentation.py`.
    [X] Ensure data is passed correctly between stages (e.g., via intermediate files or in-memory DataFrames if feasible).
    [X] Implement comprehensive logging for the overall process.
22. **Refine `setup.py` (for Deliverables)**:
    [X] Ensure `setup.py` handles all necessary setup for running the project, including environment variables and potentially invoking `start-database.sh`.
23. **Create `start-database.sh`**:
    [X] Develop a shell script to help set up or start the required database (e.g., local PostgreSQL instance or instructions for Supabase).
24. **Finalize `README.md`**:
    [X] Update `README.md` with:
        [X]Detailed instructions on how to set up the environment.
        [X]How to connect to the database (including `start-database.sh` usage).
        [X]How to run the main `process.py` script.
        [X]List of all dependencies (e.g., in a `requirements.txt` format).
25. **Install JupyterLab and Create Notebook**:
    [X] `pip install jupyterlab`.
    [X] Create a Jupyter Notebook for data exploration, analysis of results from `presentation.py`, or visualization of customer segments.
26. **(Optional) Write Project Report**:
    [X] Compose a 1-2 page report describing:
        [X]Data transformation steps and rationale.
        [X]The chosen ML model and why.
        [X]Insights gained from the ML results on the customer data.
27. **Create `Dockerfile`**:
    [ ] Develop a `Dockerfile` to containerize the application, including all dependencies and scripts.

