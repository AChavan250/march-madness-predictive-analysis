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
TEAM_FEATURES_FILE = DATA_DIR / "team_features_base.csv"

OUTPUT_FILE = OUTPUT_DIR / "bracket_predictions_2026.csv"


# ---------------------------------------------------------
# MODEL FEATURES
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


base_features = [
    "SEED",
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
]


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

games = pd.read_csv(MODEL_FILE)
team_features = pd.read_csv(TEAM_FEATURES_FILE)

train = games[
    (games["YEAR"] <= 2025)
    & games["TEAM1_WIN"].notna()
].copy()

teams_2026 = team_features[
    team_features["YEAR"] == 2026
].copy()


X_train = train[features]
y_train = train["TEAM1_WIN"].astype(int)


# ---------------------------------------------------------
# TRAIN MODELS ON ALL HISTORICAL GAMES
# ---------------------------------------------------------

rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=6,
    min_samples_leaf=5,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)


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

xgb.fit(X_train, y_train)


scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

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


# ---------------------------------------------------------
# 2026 FIRST-ROUND BRACKET
# ---------------------------------------------------------
# Order matters.
# Winners of games 1 & 2 meet next, then 3 & 4, etc.

round64 = [
    ("Duke", "Siena"),
    ("Ohio St.", "TCU"),
    ("St. John's", "Northern Iowa"),
    ("Kansas", "Cal Baptist"),

    ("Louisville", "South Florida"),
    ("Michigan St.", "North Dakota St."),
    ("UCLA", "UCF"),
    ("Connecticut", "Furman"),

    ("Florida", "Prairie View A&M"),
    ("Clemson", "Iowa"),
    ("Vanderbilt", "McNeese St."),
    ("Nebraska", "Troy"),

    ("North Carolina", "VCU"),
    ("Illinois", "Penn"),
    ("Saint Mary's", "Texas A&M"),
    ("Houston", "Idaho"),

    ("Arizona", "LIU Brooklyn"),
    ("Villanova", "Utah St."),
    ("Wisconsin", "High Point"),
    ("Arkansas", "Hawaii"),

    ("BYU", "North Carolina St."),
    ("Gonzaga", "Kennesaw St."),
    ("Miami FL", "Missouri"),
    ("Purdue", "Queens"),

    ("Michigan", "UMBC"),
    ("Georgia", "Saint Louis"),
    ("Texas Tech", "Akron"),
    ("Alabama", "Hofstra"),

    ("Tennessee", "SMU"),
    ("Virginia", "Wright St."),
    ("Kentucky", "Santa Clara"),
    ("Iowa St.", "Tennessee St."),
]


# ---------------------------------------------------------
# LOOK UP TEAM FEATURES
# ---------------------------------------------------------

def get_team(team_name):

    row = teams_2026[
        teams_2026["TEAM"] == team_name
    ]

    if row.empty:
        raise ValueError(
            f"Team not found in 2026 features: {team_name}"
        )

    return row.iloc[0]


# ---------------------------------------------------------
# BUILD ONE MATCHUP FEATURE ROW
# ---------------------------------------------------------

def build_matchup(team1, team2):

    t1 = get_team(team1)
    t2 = get_team(team2)

    values = {}

    for col in base_features:

        values[f"{col}_DIFF"] = (
            t1[col] - t2[col]
        )

    return pd.DataFrame(
        [values]
    )[features]


# ---------------------------------------------------------
# PREDICT A SINGLE GAME
# ---------------------------------------------------------

def predict_game(team1, team2, method):

    X = build_matchup(
        team1,
        team2
    )

    if method == "Random Forest":

        probability = rf.predict_proba(X)[0, 1]

    elif method == "XGBoost":

        probability = xgb.predict_proba(X)[0, 1]

    elif method == "Neural Network":

        X_scaled = scaler.transform(X)

        probability = nn.predict_proba(
            X_scaled
        )[0, 1]

    elif method == "KADJ EM":

        probability = (
            1.0
            if X.iloc[0]["KADJ EM_DIFF"] > 0
            else 0.0
        )

    elif method == "BARTHAG":

        probability = (
            1.0
            if X.iloc[0]["BARTHAG_DIFF"] > 0
            else 0.0
        )

    elif method == "TR Rating":

        probability = (
            1.0
            if X.iloc[0]["TR RATING_DIFF"] > 0
            else 0.0
        )

    else:
        raise ValueError(
            f"Unknown method: {method}"
        )

    if probability >= 0.50:
        winner = team1
        loser = team2

    else:
        winner = team2
        loser = team1

    return winner, loser, probability


# ---------------------------------------------------------
# SIMULATE FULL BRACKET
# ---------------------------------------------------------

round_names = [
    "R64",
    "R32",
    "S16",
    "E8",
    "F4",
    "Final",
]


def simulate_bracket(method):

    current_games = round64.copy()

    results = []

    for round_name in round_names:

        winners = []

        for game_number, (
            team1,
            team2
        ) in enumerate(
            current_games,
            start=1
        ):

            winner, loser, probability = (
                predict_game(
                    team1,
                    team2,
                    method
                )
            )

            winners.append(winner)

            results.append({
                "METHOD": method,
                "ROUND": round_name,
                "GAME": game_number,
                "TEAM1": team1,
                "TEAM2": team2,
                "WINNER": winner,
                "LOSER": loser,
                "TEAM1_WIN_PROBABILITY":
                    probability,
            })

        # Championship has no next round.
        if len(winners) == 1:
            break

        # Pair adjacent winners.
        current_games = [
            (
                winners[i],
                winners[i + 1]
            )
            for i in range(
                0,
                len(winners),
                2
            )
        ]

    return results


# ---------------------------------------------------------
# RUN ALL METHODS
# ---------------------------------------------------------

methods = [
    "Random Forest",
    "XGBoost",
    "Neural Network",
    "KADJ EM",
    "BARTHAG",
    "TR Rating",
]


all_results = []

for method in methods:

    all_results.extend(
        simulate_bracket(method)
    )


results_df = pd.DataFrame(
    all_results
)


# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ---------------------------------------------------------
# PRINT FINAL FOUR + CHAMPION
# ---------------------------------------------------------

print(
    "\n--- 2026 Full Bracket Predictions ---"
)


for method in methods:

    method_results = results_df[
        results_df["METHOD"] == method
    ]

    final_four_games = method_results[
        method_results["ROUND"] == "F4"
    ]

    final_four = sorted(
        set(
            final_four_games["TEAM1"].tolist()
            + final_four_games["TEAM2"].tolist()
        )
    )

    championship = method_results[
        method_results["ROUND"] == "Final"
    ].iloc[0]

    print(
        f"\n{method}"
    )

    print(
        "Final Four: "
        + ", ".join(final_four)
    )

    print(
        f"Championship: "
        f"{championship['TEAM1']} "
        f"vs {championship['TEAM2']}"
    )

    print(
        f"Champion: "
        f"{championship['WINNER']}"
    )