# Power System Load Type Classification

This project trains a machine-learning classifier to predict whether a power-system observation represents `Light_Load`, `Medium_Load`, or `Maximum_Load`. It uses a chronological validation and test strategy so model selection and final evaluation reflect prediction on later, unseen data.

## Key Features

- Cleans and validates the timestamp and target columns, and reports dataset coverage, missing values, duplicates, and class counts.
- Creates calendar and cyclical time features from the timestamp, and uses the available `NSM` measurement as an input feature.
- Preprocesses numeric and categorical inputs in scikit-learn pipelines, including imputation, scaling, and one-hot encoding where applicable.
- Compares Logistic Regression and Random Forest using macro F1 on the month immediately before the test period.
- Reserves the final observed calendar month for the final evaluation; that month is not used for model selection.
- Reports accuracy, macro precision, macro recall, macro F1, a per-class classification report, and a confusion matrix.
- Saves plots, predictions, feature importances, model comparisons, and a text summary under `outputs/`.

## Tech Stack

| Category | Technologies |
| --- | --- |
| Frontend | Not applicable; this is a command-line data science project. |
| Backend and machine learning | Python, scikit-learn |
| Data processing | pandas, NumPy |
| Database | Not applicable; the project reads a local CSV file. |
| Visualization | Matplotlib, Seaborn |
| Development tools | pip, Python virtual environment, Git (optional) |

## Project Structure

```text
assignment2_load_classification/
├── data/
│   └── load_data.csv
├── outputs/                       # Generated reports, charts, and predictions
├── load_type_classification.py    # End-to-end training and evaluation script
├── requirements.txt
└── README.md
```

## Dataset

The supplied dataset is `data/load_data.csv`. The script expects a `Date_Time` column and a `Load_Type` target column. Other columns in the included dataset provide numerical measurements, such as energy usage, reactive power, power factor, CO₂, and `NSM` (seconds from midnight).

The target classes are `Light_Load`, `Medium_Load`, and `Maximum_Load`. If you use a different dataset, keep the required column names or update `DATE_COLUMN`, `TARGET`, and `DATA_PATH` near the top of `load_type_classification.py`.

## Prerequisites

- Python 3.9 or newer
- pip
- A terminal (PowerShell on Windows, or a shell on macOS/Linux)

No database, web server, or frontend tooling is required.

## Installation and Setup

Run the following commands from the project directory.

### Windows (PowerShell)

1. Create a virtual environment:

   ```powershell
   py -m venv .venv
   ```

2. Activate it:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell prevents activation, you can run the script using `.venv\Scripts\python.exe` in the next steps instead.

3. Install the project dependencies:

   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

### macOS or Linux (optional)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run Locally

From the project root, with the virtual environment activated, run:

```powershell
python load_type_classification.py
```

On macOS or Linux, use the same command after activating the environment. The script reads `data/load_data.csv`, creates the `outputs/` directory if needed, prints the data overview and evaluation results in the terminal, and writes its generated files there. At least three distinct observed calendar months are required so the workflow can create training, validation, and final test periods.

## Validation Strategy

Data is sorted chronologically and split by observed calendar month. The penultimate observed month is used to compare models, and the last observed month is held out until the selected model is refit on all earlier data and evaluated. This avoids randomly mixing earlier and later observations and keeps the final test month out of model selection.

The primary model-selection metric is macro F1, which gives equal weight to each load class. Final test metrics and the per-class report are generated from the final month; results can vary if the dataset changes.

## Generated Outputs

The script writes these outputs under `outputs/`:

- `summary_report.txt` — selected model, time periods, and final test metrics.
- `validation_model_comparison.csv` and `validation_model_comparison.png` — model comparison on the validation month.
- `final_classification_report.csv` and `final_confusion_matrix.png` — final test evaluation.
- `load_type_predictions.csv` — timestamps, actual classes, predictions, and correctness for the test period.
- `random_forest_feature_importance.csv` and `feature_importance.png` — Random Forest feature importances.
- `load_type_distribution.png`, `usage_distribution.png`, and `usage_by_load_type.png` — exploratory plots (usage plots are generated when `Usage_kWh` is present).

These files are regenerated when the script runs. Review the current run's `summary_report.txt` and classification report when reporting model results.
