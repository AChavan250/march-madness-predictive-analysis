from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# FILE PATHS
# ---------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_DIR / "outputs"

BASE_RESULTS = OUTPUT_DIR / "base_model_results.csv"
XGB_RESULTS = OUTPUT_DIR / "xgboost_results.csv"
NN_RESULTS = OUTPUT_DIR / "neural_network_results.csv"

FINAL_RESULTS = OUTPUT_DIR / "model_comparison.csv"


# ---------------------------------------------------------
# LOAD RESULTS
# ---------------------------------------------------------

base = pd.read_csv(BASE_RESULTS)
xgb = pd.read_csv(XGB_RESULTS)
nn = pd.read_csv(NN_RESULTS)


# ---------------------------------------------------------
# KEEP COMMON METRICS
# ---------------------------------------------------------

columns = [
    "MODEL",
    "VALIDATION_ACCURACY",
    "TEST_2025_ACCURACY",
    "TEST_2025_CORRECT",
    "TEST_2025_GAMES",
]

base = base[columns]
xgb = xgb[columns]
nn = nn[columns]


# ---------------------------------------------------------
# COMBINE RESULTS
# ---------------------------------------------------------

comparison = pd.concat(
    [base, xgb, nn],
    ignore_index=True
)


# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------

comparison.to_csv(
    FINAL_RESULTS,
    index=False
)


# ---------------------------------------------------------
# DISPLAY AS PERCENTAGES
# ---------------------------------------------------------

display = comparison.copy()

display["VALIDATION_ACCURACY"] = (
    display["VALIDATION_ACCURACY"] * 100
).round(2)

display["TEST_2025_ACCURACY"] = (
    display["TEST_2025_ACCURACY"] * 100
).round(2)


print("\n--- Final Model Comparison ---")

print(
    display
    .sort_values(
        "VALIDATION_ACCURACY",
        ascending=False
    )
    .to_string(index=False)
)