# SNH-AI Project

This project implements an end-to-end data pipeline to extract, clean, transform customer data, identify customer segments using KMeans clustering, and train a predictive model for transaction counts.

## Setup & Execution

### Prerequisites

*   Python 3.9+
*   Access to a PostgreSQL database (Supabase)
*   Git

### 1. Clone Repository

```bash
git clone https://github.com/wrenchchatrepo/snh-ai.git
cd snh-ai
```

### 2. Create Environment & Install Dependencies

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate # On Windows use `.venv\Scripts\\activate`

# Install required packages
pip install -r requirements.txt
```

### 3. Configure Credentials

*   Copy or rename `.env.example` to `.env` (if `.env` doesn't exist).
    *   Note: The initial `setup.py` run (in the next step) can also create `.env.example` if needed.
*   Edit the `.env` file and provide your actual credentials for:
    *   `SUPABASE_URL`
    *   `SUPABASE_SERVICE_ROLE` (or the key name you use)
    *   `AXIOM_TOKEN`
    *   (Optional) Other keys like `GITHUB_PAT`, `ANTHROPIC_API_KEY`, `SUPABASE_DB_PASSWORD`, etc.

### 4. Initial Project Setup & Database Table Creation

*   **Run `setup.py`:** This script ensures necessary directories exist and checks for the `.env` file.
    ```bash
    python3 setup.py
    ```
*   **Create Database Tables:** Connect to your Supabase project (e.g., via the SQL Editor) and execute the `CREATE TABLE` statements found within the descriptions in `data/data_history.md` for the following tables:
    *   `raw_customer_data`
    *   `cleaned_customer_data`
    *   `transformed_customer_data`
    *   `customer_segments`
    *   `transaction_predictions`
    *   *(TODO: Automate this step using a db_setup script or Supabase migrations)*

### 5. Run Full Data Pipeline

This single command executes all data processing steps (Ingestion -> Cleaning -> Transformation -> KMeans Segmentation & Load).

```bash
# Ensure virtual environment is active
source .venv/bin/activate

# Run the main process script
python3 process.py
```
Monitor the console output for success or errors. Logs are also sent to Axiom if configured. This populates all the database tables sequentially.

### 6. Run Predictive Model Training

This script trains the RandomForestRegressor for transaction prediction and saves the model artifacts.

```bash
# Ensure virtual environment is active
source .venv/bin/activate

# Run the predictive modeling script
python3 src/predictive_model.py
```

### 7. View Analysis & Visualizations

The Jupyter Notebook analyzes the generated segments and includes visualizations.

1.  **Start JupyterLab:**
    (From the `snh-ai` directory, with the virtual environment active)
    ```bash
    jupyter lab
    ```
2.  **Open Notebook:** JupyterLab will open in your web browser. Navigate to and open `snh_analysis.ipynb`.
3.  **Run Cells:** Execute the cells sequentially (`Shift + Enter`) to see the analysis and plots.