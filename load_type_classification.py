"""
Power Load Type Classification
------------------------------
Reproducible time-aware classification workflow.

Expected input:
    data/load_data.csv

Expected columns:
    Date_Time, Usage_kWh, ... , Load_Type

Outputs are written to:
    outputs/

Important:
- The final observed calendar month is held out as the test set.
- The preceding observed month is used for model selection.
- The final test month is not used to choose the model.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# ----------------------------- Configuration -----------------------------

DATA_PATH = Path("data/load_data.csv")
OUTPUT_DIR = Path("outputs")
TARGET = "Load_Type"
DATE_COLUMN = "Date_Time"
RANDOM_STATE = 42

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Consistent visual styling for all figures.
sns.set_theme(style="whitegrid", context="notebook", palette="Set2")
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 180,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.titlesize": 13,
    "figure.titlesize": 15,
})


def section(title: str) -> None:
    """Print a readable section heading in the terminal."""
    print("\n" + "═" * 72)
    print(f"  {title}")
    print("═" * 72)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build preprocessing fitted only on the training partition."""
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

    transformers = []
    if numeric_features:
        numeric_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("numeric", numeric_pipe, numeric_features))

    if categorical_features:
        categorical_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(("categorical", categorical_pipe, categorical_features))

    if not transformers:
        raise ValueError("No usable input features were found.")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_models(X_train: pd.DataFrame) -> dict:
    """Create models with separate preprocessing instances."""
    return {
        "Random Forest": Pipeline([
            ("preprocessor", build_preprocessor(X_train)),
            ("classifier", RandomForestClassifier(
                n_estimators=250,
                random_state=RANDOM_STATE,
                class_weight="balanced",
                n_jobs=-1,
            )),
        ]),
        "Logistic Regression": Pipeline([
            ("preprocessor", build_preprocessor(X_train)),
            ("classifier", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )),
        ]),
    }


def calculate_metrics(y_true, y_pred) -> dict:
    """Return the required metrics, with macro averaging for class balance."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Macro Precision": precision_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "Macro Recall": recall_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
        "Macro F1": f1_score(
            y_true, y_pred, average="macro", zero_division=0
        ),
    }


def save_metric_chart(comparison: pd.DataFrame) -> None:
    chart_data = comparison.melt(
        id_vars="Model", var_name="Metric", value_name="Score"
    )
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.barplot(
        data=chart_data,
        x="Metric",
        y="Score",
        hue="Model",
        ax=ax,
    )
    ax.set(title="Model comparison on the validation month", xlabel="", ylabel="Score")
    ax.set_ylim(0, 1)
    ax.legend(title="Classifier", frameon=True)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "validation_model_comparison.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    section("POWER LOAD TYPE CLASSIFICATION")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH.resolve()}\n"
            "Place the dataset at data/load_data.csv or update DATA_PATH."
        )

    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.astype(str).str.strip()

    required = {DATE_COLUMN, TARGET}
    missing_required = required - set(df.columns)
    if missing_required:
        raise ValueError(
            f"Missing required column(s): {sorted(missing_required)}. "
            f"Columns found: {df.columns.tolist()}"
        )

    # Parse dates and normalize target labels.
    df[DATE_COLUMN] = pd.to_datetime(
        df[DATE_COLUMN], dayfirst=True, errors="coerce"
    )
    df[TARGET] = df[TARGET].astype("string").str.strip()
    df = df.dropna(subset=[DATE_COLUMN, TARGET]).copy()
    df = df[df[TARGET] != ""].sort_values(DATE_COLUMN).reset_index(drop=True)

    if df.empty:
        raise ValueError("No usable rows remain after date/target validation.")

    # Convert target to ordinary strings for consistent sklearn behaviour.
    df[TARGET] = df[TARGET].astype(str)

    # Create calendar/time features, including cyclical representations.
    dt = df[DATE_COLUMN]
    df["Hour"] = dt.dt.hour
    df["DayOfWeek"] = dt.dt.dayofweek
    df["DayOfMonth"] = dt.dt.day
    df["Month"] = dt.dt.month
    df["DayOfYear"] = dt.dt.dayofyear
    df["HourSin"] = np.sin(2 * np.pi * df["Hour"] / 24)
    df["HourCos"] = np.cos(2 * np.pi * df["Hour"] / 24)
    df["WeekdaySin"] = np.sin(2 * np.pi * df["DayOfWeek"] / 7)
    df["WeekdayCos"] = np.cos(2 * np.pi * df["DayOfWeek"] / 7)

    # Use observed calendar months, not hard-coded years/months.
    df["_MonthPeriod"] = df[DATE_COLUMN].dt.to_period("M")
    observed_months = sorted(df["_MonthPeriod"].unique())
    if len(observed_months) < 3:
        raise ValueError(
            "At least three observed calendar months are required: "
            "training month(s), a validation month, and a final test month."
        )

    test_month = observed_months[-1]
    validation_month = observed_months[-2]

    test_mask = df["_MonthPeriod"] == test_month
    validation_mask = df["_MonthPeriod"] == validation_month
    train_mask = df["_MonthPeriod"] < validation_month

    # Do not give the raw timestamp or helper period to the model.
    excluded = {TARGET, DATE_COLUMN, "_MonthPeriod"}
    feature_columns = [c for c in df.columns if c not in excluded]
    X = df[feature_columns].copy()
    y = df[TARGET].copy()

    X_train, y_train = X.loc[train_mask], y.loc[train_mask]
    X_val, y_val = X.loc[validation_mask], y.loc[validation_mask]
    X_test, y_test = X.loc[test_mask], y.loc[test_mask]

    if X_train.empty or X_val.empty or X_test.empty:
        raise ValueError("Temporal split produced an empty partition.")
    if y_train.nunique() < 2:
        raise ValueError("Training partition has fewer than two target classes.")

    section("DATA OVERVIEW")
    print(f"Rows used:             {len(df):,}")
    print(f"Features:              {len(feature_columns)}")
    print(f"Date range:            {df[DATE_COLUMN].min()} to {df[DATE_COLUMN].max()}")
    print(f"Target classes:        {', '.join(sorted(y.unique()))}")
    print(f"Missing values:        {int(df[feature_columns].isna().sum().sum()):,}")
    print(f"Duplicate rows:        {int(df.duplicated().sum()):,}")
    print("\nTarget distribution:")
    print(df[TARGET].value_counts().rename_axis("Load type").to_string())

    section("CHRONOLOGICAL PARTITIONS")
    print(f"Training:   months before {validation_month} | {len(X_train):,} rows")
    print(f"Validation: {validation_month}              | {len(X_val):,} rows")
    print(f"Final test: {test_month}              | {len(X_test):,} rows")
    print("The final test month is held out from model selection.")

    # EDA charts
    fig, ax = plt.subplots(figsize=(8, 4.8))
    counts = df[TARGET].value_counts().rename_axis("Load type").reset_index(name="Records")
    sns.barplot(data=counts, x="Load type", y="Records", hue="Load type",
                legend=False, ax=ax)
    ax.set_title("Distribution of load classes")
    ax.set_xlabel("")
    ax.set_ylabel("Number of records")
    ax.tick_params(axis="x", rotation=12)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "load_type_distribution.png", bbox_inches="tight")
    plt.close(fig)

    if "Usage_kWh" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 4.8))
        sns.histplot(data=df, x="Usage_kWh", bins=40, kde=True, ax=ax)
        ax.set_title("Electricity usage distribution")
        ax.set_xlabel("Usage (kWh)")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "usage_distribution.png", bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4.8))
        sns.boxplot(data=df, x=TARGET, y="Usage_kWh", hue=TARGET,
                    legend=False, ax=ax)
        ax.set_title("Electricity usage by load class")
        ax.set_xlabel("")
        ax.set_ylabel("Usage (kWh)")
        ax.tick_params(axis="x", rotation=12)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "usage_by_load_type.png", bbox_inches="tight")
        plt.close(fig)

    # Model selection uses validation month only.
    section("MODEL SELECTION")
    validation_rows = []
    for name, model in build_models(X_train).items():
        print(f"Fitting {name}...")
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        metrics = calculate_metrics(y_val, val_pred)
        validation_rows.append({"Model": name, **metrics})
        print(f"  Validation macro F1: {metrics['Macro F1']:.4f}")

    validation_results = pd.DataFrame(validation_rows).sort_values(
        "Macro F1", ascending=False
    ).reset_index(drop=True)
    validation_results.to_csv(OUTPUT_DIR / "validation_model_comparison.csv", index=False)
    save_metric_chart(validation_results)
    best_name = validation_results.iloc[0]["Model"]
    print("\nValidation results:")
    print(validation_results.to_string(
        index=False, formatters={
            "Accuracy": "{:.4f}".format,
            "Macro Precision": "{:.4f}".format,
            "Macro Recall": "{:.4f}".format,
            "Macro F1": "{:.4f}".format,
        }
    ))
    print(f"\nSelected model by validation macro F1: {best_name}")

    # Refit selected model on all data before the final test month.
    pretest_mask = df["_MonthPeriod"] < test_month
    X_pretest, y_pretest = X.loc[pretest_mask], y.loc[pretest_mask]
    final_model = build_models(X_pretest)[best_name]
    final_model.fit(X_pretest, y_pretest)
    test_predictions = final_model.predict(X_test)

    test_metrics = calculate_metrics(y_test, test_predictions)
    section(f"FINAL TEST RESULTS: {test_month}")
    print(f"Selected model: {best_name}")
    for metric, value in test_metrics.items():
        print(f"{metric:<18} {value:.4f}")

    labels = sorted(y.unique())
    report_dict = classification_report(
        y_test, test_predictions, labels=labels, output_dict=True, zero_division=0
    )
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(OUTPUT_DIR / "final_classification_report.csv", index=True)

    print("\nPer-class classification report:")
    print(classification_report(
        y_test, test_predictions, labels=labels, zero_division=0
    ))

    cm = confusion_matrix(y_test, test_predictions, labels=labels)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format="d", xticks_rotation=25, ax=ax, colorbar=False)
    ax.set_title(f"Final test confusion matrix ({test_month})")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "final_confusion_matrix.png", bbox_inches="tight")
    plt.close(fig)

    # Save timestamp, actual label, prediction and correctness indicator.
    predictions_df = pd.DataFrame({
        DATE_COLUMN: df.loc[test_mask, DATE_COLUMN].to_numpy(),
        "Actual_Load_Type": y_test.to_numpy(),
        "Predicted_Load_Type": test_predictions,
    })
    predictions_df["Correct"] = (
        predictions_df["Actual_Load_Type"] == predictions_df["Predicted_Load_Type"]
    )
    predictions_df.to_csv(OUTPUT_DIR / "load_type_predictions.csv", index=False)

    # Feature importance is available for the tree model. Fit it on all pre-test data.
    rf_model = build_models(X_pretest)["Random Forest"]
    rf_model.fit(X_pretest, y_pretest)
    rf_classifier = rf_model.named_steps["classifier"]
    fitted_preprocessor = rf_model.named_steps["preprocessor"]
    feature_names = fitted_preprocessor.get_feature_names_out()
    importances = pd.Series(
        rf_classifier.feature_importances_, index=feature_names, name="Importance"
    ).sort_values(ascending=False)
    importances.to_csv(OUTPUT_DIR / "random_forest_feature_importance.csv")

    top = importances.head(12).sort_values()
    fig, ax = plt.subplots(figsize=(9, 6))
    top.plot(kind="barh", ax=ax, color=sns.color_palette("Set2", n_colors=len(top)))
    ax.set_title("Most influential features (Random Forest)")
    ax.set_xlabel("Feature importance")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "feature_importance.png", bbox_inches="tight")
    plt.close(fig)

    summary = [
        "POWER LOAD TYPE CLASSIFICATION REPORT",
        "=" * 42,
        f"Rows used: {len(df)}",
        f"Date range: {df[DATE_COLUMN].min()} to {df[DATE_COLUMN].max()}",
        f"Training period: through {validation_month - 1}",
        f"Validation month: {validation_month}",
        f"Final test month: {test_month}",
        f"Training rows before final test: {len(X_pretest)}",
        f"Final test rows: {len(X_test)}",
        f"Selected model (validation macro F1): {best_name}",
        "",
        "Final test metrics:",
        *[f"{key}: {value:.4f}" for key, value in test_metrics.items()],
        "",
        "Output files are in the outputs/ directory.",
    ]
    (OUTPUT_DIR / "summary_report.txt").write_text("\n".join(summary), encoding="utf-8")

    section("COMPLETE")
    print(f"Results saved in: {OUTPUT_DIR.resolve()}")
    print("Review summary_report.txt, final_classification_report.csv,")
    print("load_type_predictions.csv and final_confusion_matrix.png.")


if __name__ == "__main__":
    main()
