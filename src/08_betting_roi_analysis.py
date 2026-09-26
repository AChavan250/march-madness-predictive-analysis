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
VEGAS_FILE = DATA_DIR / "vegas_lines.csv"

RESULTS_FILE = OUTPUT_DIR / "betting_roi_results.csv"
BET_DETAILS_FILE = OUTPUT_DIR / "betting_game_details.csv"


# ---------------------------------------------------------
# STEP 1: LOAD DATA
# ---------------------------------------------------------

games = pd.read_csv(MODEL_FILE)
vegas = pd.read_csv(VEGAS_FILE)


# ---------------------------------------------------------
# STEP 2: FEATURES
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
# STEP 3: CREATE MATCHUP KEYS
# ---------------------------------------------------------
# This lets us merge games even if Team 1 / Team 2
# appear in the opposite order from A / B in Vegas data.

games["LOW_TEAM_NO"] = games[
    ["TEAM1_NO", "TEAM2_NO"]
].min(axis=1)

games["HIGH_TEAM_NO"] = games[
    ["TEAM1_NO", "TEAM2_NO"]
].max(axis=1)


vegas["LOW_TEAM_NO"] = vegas[
    ["A_TEAM_NO", "B_TEAM_NO"]
].min(axis=1)

vegas["HIGH_TEAM_NO"] = vegas[
    ["A_TEAM_NO", "B_TEAM_NO"]
].max(axis=1)


# ---------------------------------------------------------
# STEP 4: MERGE VEGAS DATA
# ---------------------------------------------------------

vegas_keep = vegas[
    [
        "YEAR",
        "LOW_TEAM_NO",
        "HIGH_TEAM_NO",
        "A_TEAM",
        "B_TEAM",
        "FAVORED_TEAM_SR",
        "SPREAD",
        "SPREAD_RAW",
    ]
].copy()


data = games.merge(
    vegas_keep,
    on=[
        "YEAR",
        "LOW_TEAM_NO",
        "HIGH_TEAM_NO",
    ],
    how="left"
)


# ---------------------------------------------------------
# STEP 5: NORMALIZE TEAM NAMES
# ---------------------------------------------------------

aliases = {
    "Washington State": "Washington St.",
    "Kent State": "Kent St.",
    "Southern California": "USC",
    "Mississippi State": "Mississippi St.",
    "Michigan State": "Michigan St.",
    "Miami (FL)": "Miami FL",
    "Ohio State": "Ohio St.",
    "Virginia Commonwealth": "VCU",
    "North Carolina State": "North Carolina St.",
    "Brigham Young": "BYU",
    "Arizona State": "Arizona St.",
    "Iowa State": "Iowa St.",
    "Colorado State": "Colorado St.",
    "Florida State": "Florida St.",
    "Kansas State": "Kansas St.",
    "Wichita State": "Wichita St.",
    "Oklahoma State": "Oklahoma St.",
    "Oregon State": "Oregon St.",
    "Boise State": "Boise St.",
    "San Diego State": "San Diego St.",
    "Penn State": "Penn St.",
    "Murray State": "Murray St.",
    "Morehead State": "Morehead St.",
    "Norfolk State": "Norfolk St.",
    "Weber State": "Weber St.",
    "South Dakota State": "South Dakota St.",
    "Wright State": "Wright St.",
    "Kennesaw State": "Kennesaw St.",
    "Cleveland State": "Cleveland St.",
    "Miami (OH)": "Miami OH",
    "Nevada-Las Vegas": "UNLV",
    "Louisiana State": "LSU",
    "Southern Methodist": "SMU",
    "Central Florida": "UCF",
    "Loyola (IL)": "Loyola Chicago",
    "Cal State Fullerton": "Cal St. Fullerton",
    "Gardner-Webb": "Gardner Webb",
    "North Carolina-Wilmington": "UNC Wilmington",
    "North Carolina-Asheville": "UNC Asheville",
    "UC-Irvine": "UC Irvine",
    "UC-Santa Barbara": "UC Santa Barbara",
    "UC-San Diego": "UC San Diego",
    "Alabama-Birmingham": "UAB",
    "Arkansas-Little Rock": "Little Rock",
    "Texas Christian": "TCU",
    "Middle Tennessee State": "Middle Tennessee",
    "Georgia State": "Georgia St.",
    "Utah State": "Utah St.",
    "New Mexico State": "New Mexico St.",
    "NC State": "North Carolina St.",
    "Saint Mary's (CA)": "Saint Mary's",
    "Saint Peter's": "Saint Peter's",
    "Texas A&M-Corpus Christi": "Texas A&M Corpus Christi",
}


def clean_team_name(name):

    if pd.isna(name):
        return None

    name = str(name).strip()

    return aliases.get(
        name,
        name
    )


# ---------------------------------------------------------
# STEP 6: IDENTIFY VEGAS FAVORITE
# ---------------------------------------------------------

def vegas_pick_team1(row):

    favorite = clean_team_name(
        row["FAVORED_TEAM_SR"]
    )

    if favorite is None:
        return None

    team1 = str(row["TEAM1"]).strip()
    team2 = str(row["TEAM2"]).strip()

    if favorite == team1:
        return 1

    if favorite == team2:
        return 0

    # fallback for minor naming differences
    if (
        favorite.lower() in team1.lower()
        or team1.lower() in favorite.lower()
    ):
        return 1

    if (
        favorite.lower() in team2.lower()
        or team2.lower() in favorite.lower()
    ):
        return 0

    return None


data["VEGAS_PICK_TEAM1"] = data.apply(
    vegas_pick_team1,
    axis=1
)


# ---------------------------------------------------------
# STEP 7: TRAIN / OOS SPLIT
# ---------------------------------------------------------

train = data[
    data["YEAR"].between(2008, 2019)
].copy()

oos = data[
    data["YEAR"].between(2021, 2025)
].copy()


X_train = train[features]
y_train = train["TEAM1_WIN"].astype(int)

X_oos = oos[features]


# ---------------------------------------------------------
# STEP 8: RANDOM FOREST
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

oos["RF_PRED"] = rf.predict(
    X_oos
)


# ---------------------------------------------------------
# STEP 9: XGBOOST
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

oos["XGB_PRED"] = xgb.predict(
    X_oos
)


# ---------------------------------------------------------
# STEP 10: NEURAL NETWORK
# ---------------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_oos_scaled = scaler.transform(
    X_oos
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

oos["NN_PRED"] = nn.predict(
    X_oos_scaled
)


# ---------------------------------------------------------
# STEP 11: BASELINE PREDICTIONS
# ---------------------------------------------------------

oos["KADJ_EM_PRED"] = (
    oos["KADJ EM_DIFF"] > 0
).astype(int)

oos["TR_RATING_PRED"] = (
    oos["TR RATING_DIFF"] > 0
).astype(int)


# ---------------------------------------------------------
# STEP 12: ESTIMATED UNDERDOG PAYOUT
# ---------------------------------------------------------
# Approximate profit from a $100 winning underdog bet.
#
# Example:
# payout = 220 means a winning $100 bet earns $220 profit.
#
# These are approximations derived from point spreads,
# not actual historical moneyline prices.

def estimated_underdog_profit(spread):

    s = abs(spread)

    if s <= 1:
        return 105

    if s <= 2.5:
        return 120

    if s <= 4:
        return 140

    if s <= 6:
        return 175

    if s <= 8:
        return 220

    if s <= 10:
        return 270

    if s <= 14:
        return 350

    if s <= 20:
        return 500

    return 800


# ---------------------------------------------------------
# STEP 13: KEEP VALID VEGAS GAMES
# ---------------------------------------------------------

betting_data = oos[
    oos["SPREAD"].notna()
    & oos["VEGAS_PICK_TEAM1"].notna()
    & oos["TEAM1_WIN"].notna()
].copy()

betting_data["VEGAS_PICK_TEAM1"] = (
    betting_data["VEGAS_PICK_TEAM1"]
    .astype(int)
)

betting_data["TEAM1_WIN"] = (
    betting_data["TEAM1_WIN"]
    .astype(int)
)


# ---------------------------------------------------------
# STEP 14: VEGAS ACCURACY
# ---------------------------------------------------------

betting_data["VEGAS_CORRECT"] = (
    betting_data["VEGAS_PICK_TEAM1"]
    == betting_data["TEAM1_WIN"]
)


# ---------------------------------------------------------
# STEP 15: BETTING FUNCTION
# ---------------------------------------------------------

def calculate_roi(
    df,
    prediction_column,
    model_name
):

    # Only bet when model disagrees with Vegas.
    bets = df[
        df[prediction_column]
        != df["VEGAS_PICK_TEAM1"]
    ].copy()

    wins = 0
    profit = 0

    bet_results = []

    for _, row in bets.iterrows():

        correct = (
            row[prediction_column]
            == row["TEAM1_WIN"]
        )

        if correct:

            payout = estimated_underdog_profit(
                row["SPREAD"]
            )

            profit += payout
            wins += 1

            game_profit = payout

        else:

            profit -= 100
            game_profit = -100

        bet_results.append({
            "MODEL": model_name,
            "YEAR": row["YEAR"],
            "TEAM1": row["TEAM1"],
            "TEAM2": row["TEAM2"],
            "SPREAD": row["SPREAD"],
            "VEGAS_PICK_TEAM1":
                row["VEGAS_PICK_TEAM1"],
            "MODEL_PICK_TEAM1":
                row[prediction_column],
            "ACTUAL_TEAM1_WIN":
                row["TEAM1_WIN"],
            "BET_WON": correct,
            "PROFIT": game_profit,
        })

    number_bets = len(bets)

    amount_wagered = (
        number_bets * 100
    )

    if number_bets > 0:

        roi = (
            profit
            / amount_wagered
        ) * 100

        win_rate = (
            wins
            / number_bets
        ) * 100

    else:

        roi = 0
        win_rate = 0

    summary = {
        "MODEL": model_name,
        "BETS": number_bets,
        "WINS": wins,
        "WIN_RATE": win_rate,
        "TOTAL_WAGERED": amount_wagered,
        "PROFIT": profit,
        "ROI": roi,
    }

    return summary, bet_results


# ---------------------------------------------------------
# STEP 16: RUN BETTING STRATEGIES
# ---------------------------------------------------------

models = {
    "Random Forest": "RF_PRED",
    "XGBoost": "XGB_PRED",
    "Neural Network": "NN_PRED",
    "KADJ EM": "KADJ_EM_PRED",
    "TR Rating": "TR_RATING_PRED",
}


summaries = []
all_bets = []


for model_name, prediction_column in models.items():

    summary, details = calculate_roi(
        betting_data,
        prediction_column,
        model_name
    )

    summaries.append(summary)
    all_bets.extend(details)


results = pd.DataFrame(
    summaries
)

bet_details = pd.DataFrame(
    all_bets
)


# ---------------------------------------------------------
# STEP 17: SAVE OUTPUTS
# ---------------------------------------------------------

results.to_csv(
    RESULTS_FILE,
    index=False
)

bet_details.to_csv(
    BET_DETAILS_FILE,
    index=False
)


# ---------------------------------------------------------
# STEP 18: PRINT RESULTS
# ---------------------------------------------------------

print(
    "\n--- Vegas Data Merge ---"
)

print(
    f"Historical/OOS games 2021-2025: "
    f"{len(oos)}"
)

print(
    f"Games matched to Vegas spreads: "
    f"{oos['SPREAD'].notna().sum()}"
)

print(
    f"Vegas favorites identified: "
    f"{oos['VEGAS_PICK_TEAM1'].notna().sum()}"
)

print(
    f"Games usable for betting: "
    f"{len(betting_data)}"
)


print(
    "\n--- Vegas Favorite Accuracy ---"
)

print(
    f"{betting_data['VEGAS_CORRECT'].mean() * 100:.2f}%"
)


print(
    "\n--- Hypothetical $100 Underdog Betting ---"
)

display_results = results.copy()

display_results["WIN_RATE"] = (
    display_results["WIN_RATE"]
    .round(2)
)

display_results["ROI"] = (
    display_results["ROI"]
    .round(2)
)

print(
    display_results.to_string(
        index=False
    )
)