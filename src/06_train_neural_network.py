from pathlib import Path

import pandas as pd

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score


# ---------------------------------------------------------
# FILE PATHS
# ---------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_DATA_FILE = DATA_DIR / "model_dataset_base.csv"
RESULTS_FILE = OUTPUT_DIR / "neural_network_results.csv"


# ---------------------------------------------------------
# STEP 1: LOAD DATA
# ---------------------------------------------------------

df = pd.read_csv(MODEL_DATA_FILE)


# ---------------------------------------------------------
# STEP 2: SAME FEATURES AS RF AND XGBOOST
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
# STEP 4: BUILD NEURAL NETWORK PIPELINE
# ---------------------------------------------------------
# Neural networks are sensitive to feature scale,
# so StandardScaler is applied before training.

nn_model = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "model",
        MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            solver="adam",
            alpha=0.01,
            learning_rate_init=0.001,
            max_iter=3000,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=50,
            random_state=42
        )
    )
])


# ---------------------------------------------------------
# STEP 5: TRAIN MODEL
# ---------------------------------------------------------

nn_model.fit(
    X_train,
    y_train
)


# ---------------------------------------------------------
# STEP 6: VALIDATION PREDICTIONS
# ---------------------------------------------------------

validation_predictions = nn_model.predict(
    X_validation
)

validation_probabilities = nn_model.predict_proba(
    X_validation
)[:, 1]


# ---------------------------------------------------------
# STEP 7: TEST PREDICTIONS
# ---------------------------------------------------------

test_predictions = nn_model.predict(
    X_test
)

test_probabilities = nn_model.predict_proba(
    X_test
)[:, 1]


# ---------------------------------------------------------
# STEP 8: CALCULATE METRICS
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
# STEP 9: SAVE RESULTS
# ---------------------------------------------------------

results = pd.DataFrame([
    {
        "MODEL": "Neural Network",
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
# STEP 10: TRAINING INFORMATION
# ---------------------------------------------------------

trained_model = nn_model.named_steps["model"]


# ---------------------------------------------------------
# STEP 11: PRINT RESULTS
# ---------------------------------------------------------

print("\n--- Neural Network Dataset Split ---")

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


print("\n--- Neural Network Results ---")

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


print("\n--- Neural Network ROC-AUC ---")

print(
    f"Validation ROC-AUC: "
    f"{validation_auc:.3f}"
)

print(
    f"2025 Test ROC-AUC: "
    f"{test_auc:.3f}"
)


print("\n--- Neural Network Training ---")

print(
    f"Iterations completed: "
    f"{trained_model.n_iter_}"
)

print(
    f"Final training loss: "
    f"{trained_model.loss_:.4f}"
)