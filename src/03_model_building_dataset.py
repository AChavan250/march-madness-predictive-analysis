from pathlib import Path
import pandas as pd


# ---------------------------------------------------------
# FILE PATHS
# ---------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"

GAMES_FILE = DATA_DIR / "tournament_games_base.csv"
FEATURES_FILE = DATA_DIR / "team_features_base.csv"

OUTPUT_FILE = DATA_DIR / "model_dataset_base.csv"


# ---------------------------------------------------------
# STEP 1: LOAD DATA
# ---------------------------------------------------------

games = pd.read_csv(GAMES_FILE)
features = pd.read_csv(FEATURES_FILE)


# ---------------------------------------------------------
# STEP 2: FEATURES WE WANT TO COMPARE
# ---------------------------------------------------------

feature_columns = [
    "KADJ EM",
    "KADJ O",
    "KADJ D",
    "BADJ EM",
    "BARTHAG",
    "EFG%",
    "EFG%D",
    "TOV%",
    "TOV%D",
    "OREB%",
    "DREB%",
    "EXP",
    "TALENT",
    "ELITE SOS",
    "WAB",
    "TR RATING",
    "ELO",
    "B POWER",
    "Q1 PLUS Q2 W",
    "R SCORE",
    "AP RANK",
    "AP VOTES",
]


# ---------------------------------------------------------
# STEP 3: PREPARE TEAM 1 FEATURES
# ---------------------------------------------------------

team1_features = features[
    ["YEAR", "TEAM NO"] + feature_columns
].copy()

team1_features = team1_features.rename(
    columns={
        "TEAM NO": "TEAM1_NO",
        **{
            col: f"TEAM1_{col}"
            for col in feature_columns
        }
    }
)


# ---------------------------------------------------------
# STEP 4: PREPARE TEAM 2 FEATURES
# ---------------------------------------------------------

team2_features = features[
    ["YEAR", "TEAM NO"] + feature_columns
].copy()

team2_features = team2_features.rename(
    columns={
        "TEAM NO": "TEAM2_NO",
        **{
            col: f"TEAM2_{col}"
            for col in feature_columns
        }
    }
)


# ---------------------------------------------------------
# STEP 5: MERGE BOTH TEAMS INTO EACH GAME
# ---------------------------------------------------------

model_data = games.merge(
    team1_features,
    on=["YEAR", "TEAM1_NO"],
    how="left"
)

model_data = model_data.merge(
    team2_features,
    on=["YEAR", "TEAM2_NO"],
    how="left"
)

# ---------------------------------------------------------
# STEP 6: CREATE DIFFERENTIAL FEATURES
# ---------------------------------------------------------

for col in feature_columns:

    team1_col = f"TEAM1_{col}"
    team2_col = f"TEAM2_{col}"

    model_data[f"{col}_DIFF"] = (
        model_data[team1_col]
        - model_data[team2_col]
    )


# SEED already exists in the matchup dataset
model_data["SEED_DIFF"] = (
    model_data["TEAM1_SEED"]
    - model_data["TEAM2_SEED"]
)

# ---------------------------------------------------------
# STEP 7: CREATE TARGET VARIABLE
# ---------------------------------------------------------
# 1 = Team 1 won
# 0 = Team 2 won
# NaN = outcome unknown (2026)

model_data["TEAM1_WIN"] = pd.NA

known_games = model_data["WINNER_NO"].notna()

model_data.loc[
    known_games,
    "TEAM1_WIN"
] = (
    model_data.loc[
        known_games,
        "WINNER_NO"
    ]
    == model_data.loc[
        known_games,
        "TEAM1_NO"
    ]
).astype(int)


# ---------------------------------------------------------
# STEP 8: CHECK FOR FAILED TEAM MERGES
# ---------------------------------------------------------

team1_missing = model_data["TEAM1_KADJ EM"].isna().sum()
team2_missing = model_data["TEAM2_KADJ EM"].isna().sum()


# ---------------------------------------------------------
# STEP 9: SAVE DATA
# ---------------------------------------------------------

model_data.to_csv(
    OUTPUT_FILE,
    index=False
)


# ---------------------------------------------------------
# STEP 10: QUALITY CHECKS
# ---------------------------------------------------------

historical = model_data[
    model_data["YEAR"] <= 2025
]

prediction_2026 = model_data[
    model_data["YEAR"] == 2026
]


print("\n--- ML-Ready Base Dataset ---")

print(f"Rows: {len(model_data)}")
print(f"Columns: {len(model_data.columns)}")

print(
    f"Historical games: "
    f"{len(historical)}"
)

print(
    f"2026 games: "
    f"{len(prediction_2026)}"
)

print(
    f"Team 1 feature merge failures: "
    f"{team1_missing}"
)

print(
    f"Team 2 feature merge failures: "
    f"{team2_missing}"
)

print(
    f"Known targets: "
    f"{model_data['TEAM1_WIN'].notna().sum()}"
)

print("\nTarget Distribution:")

print(
    historical["TEAM1_WIN"]
    .value_counts(dropna=False)
)


print("\nMissing Differential Features:")

diff_columns = [
    f"{col}_DIFF"
    for col in feature_columns
]

diff_columns.append("SEED_DIFF")

print(
    model_data[diff_columns]
    .isna()
    .sum()
    .sort_values(ascending=False)
)


print("\nSample Differential Features:")

sample_cols = [
    "YEAR",
    "TEAM1",
    "TEAM2",
    "KADJ EM_DIFF",
    "BARTHAG_DIFF",
    "TR RATING_DIFF",
    "SEED_DIFF",
    "TEAM1_WIN",
]

print(
    model_data[sample_cols]
    .head()
    .to_string(index=False)
)