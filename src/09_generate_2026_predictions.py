from pathlib import Path

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


# ---------------------------------------------------------
# FILE PATHS
# ---------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"

MODEL_FILE = DATA_DIR / "model_dataset_base.csv"

OUTPUT_FILE = OUTPUT_DIR / "predictions_2026.csv"


# ---------------------------------------------------------
# STEP 1: LOAD DATA
# ---------------------------------------------------------

df = pd.read_csv(MODEL_FILE)


# ---------------------------------------------------------
# STEP 2: MODEL FEATURES
# ---------------------------------------------------------

features = [
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
# STEP 3: TRAIN ON ALL COMPLETED HISTORICAL GAMES
# ---------------------------------------------------------
# We already evaluated the models previously.
# For final 2026 predictions, we can now retrain using
# every completed tournament from 2008 through 2025.

train = df[
    (df["YEAR"] <= 2025)
    & (df["TEAM1_WIN"].notna())
].copy()

predict_2026 = df[
    df["YEAR"] == 2026
].copy()


X_train = train[features]
y_train = train["TEAM1_WIN"].astype(int)

X_2026 = predict_2026[features]


# ---------------------------------------------------------
# STEP 4: RANDOM FOREST
# ---------------------------------------------------------

rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=6,
    min_samples_leaf=5,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)

rf.fit(
    X_train,
    y_train
)

predict_2026["RF_PROB_TEAM1"] = (
    rf.predict_proba(X_2026)[:, 1]
)

predict_2026["RF_PICK_TEAM1"] = (
    predict_2026["RF_PROB_TEAM1"] >= 0.50
).astype(int)


# ---------------------------------------------------------
# STEP 5: XGBOOST
# ---------------------------------------------------------

xgb = XGBClassifier(
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

xgb.fit(
    X_train,
    y_train
)

predict_2026["XGB_PROB_TEAM1"] = (
    xgb.predict_proba(X_2026)[:, 1]
)

predict_2026["XGB_PICK_TEAM1"] = (
    predict_2026["XGB_PROB_TEAM1"] >= 0.50
).astype(int)


# ---------------------------------------------------------
# STEP 6: NEURAL NETWORK
# ---------------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_2026_scaled = scaler.transform(
    X_2026
)


nn = MLPClassifier(
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

nn.fit(
    X_train_scaled,
    y_train
)

predict_2026["NN_PROB_TEAM1"] = (
    nn.predict_proba(X_2026_scaled)[:, 1]
)

predict_2026["NN_PICK_TEAM1"] = (
    predict_2026["NN_PROB_TEAM1"] >= 0.50
).astype(int)


# ---------------------------------------------------------
# STEP 7: SIMPLE RATING PREDICTIONS
# ---------------------------------------------------------

predict_2026["KADJ_EM_PICK_TEAM1"] = (
    predict_2026["KADJ EM_DIFF"] > 0
).astype(int)

predict_2026["BARTHAG_PICK_TEAM1"] = (
    predict_2026["BARTHAG_DIFF"] > 0
).astype(int)

predict_2026["TR_RATING_PICK_TEAM1"] = (
    predict_2026["TR RATING_DIFF"] > 0
).astype(int)


# ---------------------------------------------------------
# STEP 8: CONVERT 0/1 INTO TEAM NAMES
# ---------------------------------------------------------

def picked_team(row, column):

    if row[column] == 1:
        return row["TEAM1"]

    return row["TEAM2"]


pick_columns = {
    "RF_PICK_TEAM1": "RF_PICK",
    "XGB_PICK_TEAM1": "XGB_PICK",
    "NN_PICK_TEAM1": "NN_PICK",
    "KADJ_EM_PICK_TEAM1": "KADJ_EM_PICK",
    "BARTHAG_PICK_TEAM1": "BARTHAG_PICK",
    "TR_RATING_PICK_TEAM1": "TR_RATING_PICK",
}


for source_column, output_column in pick_columns.items():

    predict_2026[output_column] = (
        predict_2026.apply(
            lambda row: picked_team(
                row,
                source_column
            ),
            axis=1
        )
    )


# ---------------------------------------------------------
# STEP 9: CREATE AVERAGE ML PROBABILITY
# ---------------------------------------------------------
# This is not declaring one model "best".
# It simply averages the three ML probability estimates.

predict_2026["ML_AVG_PROB_TEAM1"] = (
    predict_2026[
        [
            "RF_PROB_TEAM1",
            "XGB_PROB_TEAM1",
            "NN_PROB_TEAM1",
        ]
    ]
    .mean(axis=1)
)


predict_2026["ML_AVG_PICK"] = (
    predict_2026.apply(
        lambda row:
            row["TEAM1"]
            if row["ML_AVG_PROB_TEAM1"] >= 0.50
            else row["TEAM2"],
        axis=1
    )
)


# ---------------------------------------------------------
# STEP 10: SAVE USEFUL OUTPUT
# ---------------------------------------------------------

output_columns = [
    "GAME_ID",
    "CURRENT_ROUND",
    "TEAM1",
    "TEAM1_SEED",
    "TEAM2",
    "TEAM2_SEED",

    "RF_PROB_TEAM1",
    "RF_PICK",

    "XGB_PROB_TEAM1",
    "XGB_PICK",

    "NN_PROB_TEAM1",
    "NN_PICK",

    "KADJ_EM_PICK",
    "BARTHAG_PICK",
    "TR_RATING_PICK",

    "ML_AVG_PROB_TEAM1",
    "ML_AVG_PICK",
]


results = predict_2026[
    output_columns
].copy()


results.to_csv(
    OUTPUT_FILE,
    index=False
)


# ---------------------------------------------------------
# STEP 11: PRINT RESULTS
# ---------------------------------------------------------

print("\n--- 2026 Prediction Dataset ---")

print(
    f"Historical games used for training: "
    f"{len(train)}"
)

print(
    f"Potential 2026 matchups scored: "
    f"{len(results)}"
)


print("\n--- Potential Matchups by Round ---")

print(
    results
    .groupby("CURRENT_ROUND")
    .size()
)


print("\n--- Round of 64 Predictions ---")

round64 = results[
    results["CURRENT_ROUND"] == 64
]

print(
    round64[
        [
            "TEAM1",
            "TEAM2",
            "RF_PICK",
            "XGB_PICK",
            "NN_PICK",
            "KADJ_EM_PICK",
            "TR_RATING_PICK",
            "ML_AVG_PICK",
        ]
    ]
    .to_string(index=False)
)