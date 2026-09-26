from pathlib import Path

import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score


# ---------------------------------------------------------
# FILE PATHS
# ---------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_DATA_FILE = DATA_DIR / "model_dataset_base.csv"

RESULTS_FILE = OUTPUT_DIR / "xgboost_results.csv"
IMPORTANCE_FILE = OUTPUT_DIR / "xgboost_feature_importance.csv"


# ---------------------------------------------------------
# STEP 1: LOAD DATA
# ---------------------------------------------------------

df = pd.read_csv(MODEL_DATA_FILE)


# ---------------------------------------------------------
# STEP 2: USE SAME FEATURES AS RANDOM FOREST
# ---------------------------------------------------------

feature_names = [
    "SEED_DIFF",
    "KADJ EM_DIFF",
    "KADJ O_DIFF",
    "KADJ D_DIFF",
    "BADJ EM_DIFF",
    "BARTHAG_DIFF",
    "EFG%_DIFF",
    "EFG%D_DIFF",
    "TOV%_DIFF",
    "TOV%D_DIFF",
    "OREB%_DIFF",
    "DREB%_DIFF",
    "EXP_DIFF",
    "TALENT_DIFF",
    "ELITE SOS_DIFF",
    "WAB_DIFF",
    "TR RATING_DIFF",
    "ELO_DIFF",
    "B POWER_DIFF",
    "Q1 PLUS Q2 W_DIFF",
    "R SCORE_DIFF",
]


# ---------------------------------------------------------
# STEP 3: SAME TIME-BASED SPLIT
# ---------------------------------------------------------

train = df[
    (df["YEAR"] >= 2008)
    & (df["YEAR"] <= 2019)
].copy()

validation = df[
    (df["YEAR"] >= 2021)
    & (df["YEAR"] <= 2024)
].copy()

test = df[
    df["YEAR"] == 2025
].copy()


X_train = train[feature_names]
y_train = train["TEAM1_WIN"].astype(int)

X_validation = validation[feature_names]
y_validation = validation["TEAM1_WIN"].astype(int)

X_test = test[feature_names]
y_test = test["TEAM1_WIN"].astype(int)


# ---------------------------------------------------------
# STEP 4: BUILD XGBOOST MODEL
# ---------------------------------------------------------

xgb_model = XGBClassifier(
    n_estimators=400,
    max_depth=3,
    learning_rate=0.03,
    subsample=0.80,
    colsample_bytree=0.80,
    reg_alpha=0.10,
    reg_lambda=1.0,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)


# ---------------------------------------------------------
# STEP 5: TRAIN MODEL
# ---------------------------------------------------------

xgb_model.fit(
    X_train,
    y_train
)


# ---------------------------------------------------------
# STEP 6: VALIDATION PREDICTIONS
# ---------------------------------------------------------

validation_predictions = xgb_model.predict(
    X_validation
)

validation_probabilities = xgb_model.predict_proba(
    X_validation
)[:, 1]


# ---------------------------------------------------------
# STEP 7: 2025 TEST PREDICTIONS
# ---------------------------------------------------------

test_predictions = xgb_model.predict(
    X_test
)

test_probabilities = xgb_model.predict_proba(
    X_test
)[:, 1]


# ---------------------------------------------------------
# STEP 8: METRICS
# ---------------------------------------------------------

validation_accuracy = accuracy_score(
    y_validation,
    validation_predictions
)

test_accuracy = accuracy_score(
    y_test,
    test_predictions
)

validation_auc = roc_auc_score(
    y_validation,
    validation_probabilities
)

test_auc = roc_auc_score(
    y_test,
    test_probabilities
)


# ---------------------------------------------------------
# STEP 9: SAVE MODEL RESULTS
# ---------------------------------------------------------

results = pd.DataFrame([
    {
        "MODEL": "XGBoost",
        "VALIDATION_ACCURACY": validation_accuracy,
        "TEST_2025_ACCURACY": test_accuracy,
        "VALIDATION_ROC_AUC": validation_auc,
        "TEST_2025_ROC_AUC": test_auc,
        "TEST_2025_CORRECT": int(
            (test_predictions == y_test).sum()
        ),
        "TEST_2025_GAMES": len(test)
    }
])

results.to_csv(
    RESULTS_FILE,
    index=False
)


# ---------------------------------------------------------
# STEP 10: FEATURE IMPORTANCE
# ---------------------------------------------------------

feature_importance = pd.DataFrame({
    "FEATURE": feature_names,
    "IMPORTANCE": xgb_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    "IMPORTANCE",
    ascending=False
).reset_index(drop=True)

feature_importance.to_csv(
    IMPORTANCE_FILE,
    index=False
)


# ---------------------------------------------------------
# STEP 11: PRINT RESULTS
# ---------------------------------------------------------

print("\n--- XGBoost Dataset Split ---")

print(
    f"Training:   {len(train)} games "
    f"(2008-2019)"
)

print(
    f"Validation: {len(validation)} games "
    f"(2021-2024)"
)

print(
    f"Test:       {len(test)} games "
    f"(2025)"
)


print("\n--- XGBoost Results ---")

print(
    f"Validation Accuracy: "
    f"{validation_accuracy * 100:.2f}%"
)

print(
    f"2025 Test Accuracy:  "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"2025 Correct: "
    f"{(test_predictions == y_test).sum()} "
    f"/ {len(test)}"
)


print("\n--- XGBoost ROC-AUC ---")

print(
    f"Validation ROC-AUC: "
    f"{validation_auc:.3f}"
)

print(
    f"2025 Test ROC-AUC: "
    f"{test_auc:.3f}"
)


print("\n--- Top 10 XGBoost Features ---")

print(
    feature_importance
    .head(10)
    .to_string(index=False)
)