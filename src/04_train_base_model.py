from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score


# ---------------------------------------------------------
# FILE PATHS
# ---------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_DATA_FILE = DATA_DIR / "model_dataset_base.csv"

RESULTS_FILE = OUTPUT_DIR / "base_model_results.csv"
IMPORTANCE_FILE = OUTPUT_DIR / "random_forest_feature_importance.csv"


# ---------------------------------------------------------
# STEP 1: LOAD MODEL DATA
# ---------------------------------------------------------

df = pd.read_csv(MODEL_DATA_FILE)


# ---------------------------------------------------------
# STEP 2: SELECT CLEAN DIFFERENTIAL FEATURES
# ---------------------------------------------------------
# AP RANK and AP VOTES are excluded because they contain
# missing values for many tournament teams.

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
# STEP 3: TIME-BASED SPLIT
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
# STEP 4: SIMPLE BASELINE MODELS
# ---------------------------------------------------------

def chalk_prediction(data):
    """
    Lower seed number = stronger seed.

    If seeds are equal, use KADJ EM as the tie-breaker.
    """

    predictions = []

    for _, row in data.iterrows():

        if row["SEED_DIFF"] < 0:
            predictions.append(1)

        elif row["SEED_DIFF"] > 0:
            predictions.append(0)

        else:
            predictions.append(
                int(row["KADJ EM_DIFF"] > 0)
            )

    return predictions


def rating_prediction(data, column):
    """
    Positive rating difference means Team 1 has
    the stronger rating.
    """

    return (
        data[column] > 0
    ).astype(int)


# ---------------------------------------------------------
# STEP 5: TRAIN RANDOM FOREST
# ---------------------------------------------------------

rf_model = RandomForestClassifier(
    n_estimators=500,
    max_depth=6,
    min_samples_leaf=5,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)

rf_model.fit(
    X_train,
    y_train
)


# ---------------------------------------------------------
# STEP 6: EVALUATION FUNCTION
# ---------------------------------------------------------

results = []


def evaluate_baseline(
    model_name,
    validation_predictions,
    test_predictions
):

    validation_accuracy = accuracy_score(
        y_validation,
        validation_predictions
    )

    test_accuracy = accuracy_score(
        y_test,
        test_predictions
    )

    results.append({
        "MODEL": model_name,
        "VALIDATION_ACCURACY": validation_accuracy,
        "TEST_2025_ACCURACY": test_accuracy,
        "TEST_2025_CORRECT": int(
            test_accuracy * len(test)
        ),
        "TEST_2025_GAMES": len(test)
    })


# ---------------------------------------------------------
# STEP 7: EVALUATE BASELINES
# ---------------------------------------------------------

evaluate_baseline(
    "Chalk / Seed",
    chalk_prediction(validation),
    chalk_prediction(test)
)

evaluate_baseline(
    "KADJ EM",
    rating_prediction(
        validation,
        "KADJ EM_DIFF"
    ),
    rating_prediction(
        test,
        "KADJ EM_DIFF"
    )
)

evaluate_baseline(
    "BARTHAG",
    rating_prediction(
        validation,
        "BARTHAG_DIFF"
    ),
    rating_prediction(
        test,
        "BARTHAG_DIFF"
    )
)

evaluate_baseline(
    "TR RATING",
    rating_prediction(
        validation,
        "TR RATING_DIFF"
    ),
    rating_prediction(
        test,
        "TR RATING_DIFF"
    )
)


# ---------------------------------------------------------
# STEP 8: EVALUATE RANDOM FOREST
# ---------------------------------------------------------

rf_validation_predictions = rf_model.predict(
    X_validation
)

rf_validation_probabilities = rf_model.predict_proba(
    X_validation
)[:, 1]


rf_test_predictions = rf_model.predict(
    X_test
)

rf_test_probabilities = rf_model.predict_proba(
    X_test
)[:, 1]


rf_validation_accuracy = accuracy_score(
    y_validation,
    rf_validation_predictions
)

rf_test_accuracy = accuracy_score(
    y_test,
    rf_test_predictions
)

rf_validation_auc = roc_auc_score(
    y_validation,
    rf_validation_probabilities
)

rf_test_auc = roc_auc_score(
    y_test,
    rf_test_probabilities
)


results.append({
    "MODEL": "Random Forest",
    "VALIDATION_ACCURACY": rf_validation_accuracy,
    "TEST_2025_ACCURACY": rf_test_accuracy,
    "TEST_2025_CORRECT": int(
        (rf_test_predictions == y_test).sum()
    ),
    "TEST_2025_GAMES": len(test)
})


# ---------------------------------------------------------
# STEP 9: FEATURE IMPORTANCE
# ---------------------------------------------------------

feature_importance = pd.DataFrame({
    "FEATURE": feature_names,
    "IMPORTANCE": rf_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    "IMPORTANCE",
    ascending=False
).reset_index(drop=True)


# ---------------------------------------------------------
# STEP 10: SAVE RESULTS
# ---------------------------------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    RESULTS_FILE,
    index=False
)

feature_importance.to_csv(
    IMPORTANCE_FILE,
    index=False
)


# ---------------------------------------------------------
# STEP 11: PRINT RESULTS
# ---------------------------------------------------------

print("\n--- Dataset Split ---")

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


print("\n--- Base Model Results ---")

display_results = results_df.copy()

display_results["VALIDATION_ACCURACY"] = (
    display_results["VALIDATION_ACCURACY"] * 100
).round(2)

display_results["TEST_2025_ACCURACY"] = (
    display_results["TEST_2025_ACCURACY"] * 100
).round(2)

print(
    display_results.to_string(index=False)
)


print("\n--- Random Forest ROC-AUC ---")

print(
    f"Validation ROC-AUC: "
    f"{rf_validation_auc:.3f}"
)

print(
    f"2025 Test ROC-AUC: "
    f"{rf_test_auc:.3f}"
)


print("\n--- Top 10 Random Forest Features ---")

print(
    feature_importance
    .head(10)
    .to_string(index=False)
)