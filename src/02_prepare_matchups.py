from pathlib import Path
import pandas as pd


# ---------------------------------------------------------
# FILE PATHS
# ---------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"

MATCHUPS_FILE = DATA_DIR / "Tournament Matchups.csv"
OUTPUT_FILE = DATA_DIR / "tournament_games_base.csv"


# ---------------------------------------------------------
# STEP 1: LOAD MATCHUP DATA
# ---------------------------------------------------------

matchups = pd.read_csv(MATCHUPS_FILE)

print("\n--- Original Tournament Matchup Data ---")
print(f"Rows: {len(matchups)}")
print(f"Columns: {len(matchups.columns)}")


# ---------------------------------------------------------
# STEP 2: SORT DATA
# ---------------------------------------------------------
# The source dataset stores two consecutive rows per game.
# BY YEAR NO provides the ordering within each season.

matchups = matchups.sort_values(
    ["YEAR", "BY YEAR NO"],
    ascending=[True, False]
).reset_index(drop=True)


# ---------------------------------------------------------
# STEP 3: PAIR EVERY TWO ROWS INTO ONE GAME
# ---------------------------------------------------------

games = []

for year, year_data in matchups.groupby("YEAR"):

    year_data = year_data.reset_index(drop=True)

    # Each game must contain exactly two team rows.
    if len(year_data) % 2 != 0:
        print(
            f"Warning: {year} contains an odd number "
            f"of matchup rows."
        )

    for i in range(0, len(year_data) - 1, 2):

        team1 = year_data.iloc[i]
        team2 = year_data.iloc[i + 1]

        game_number = (i // 2) + 1

        game = {
            "YEAR": year,

            "GAME_ID": f"{year}_{game_number}",

            "CURRENT_ROUND": team1["CURRENT ROUND"],

            "TEAM1_NO": team1["TEAM NO"],
            "TEAM1": team1["TEAM"],
            "TEAM1_SEED": team1["SEED"],
            "TEAM1_SCORE": team1["SCORE"],

            "TEAM2_NO": team2["TEAM NO"],
            "TEAM2": team2["TEAM"],
            "TEAM2_SEED": team2["SEED"],
            "TEAM2_SCORE": team2["SCORE"],
        }

        # Historical games have known scores.
        if (
            pd.notna(team1["SCORE"])
            and pd.notna(team2["SCORE"])
        ):

            if team1["SCORE"] > team2["SCORE"]:
                game["WINNER_NO"] = team1["TEAM NO"]
                game["WINNER"] = team1["TEAM"]

            elif team2["SCORE"] > team1["SCORE"]:
                game["WINNER_NO"] = team2["TEAM NO"]
                game["WINNER"] = team2["TEAM"]

            else:
                game["WINNER_NO"] = None
                game["WINNER"] = None

        # 2026 games have not necessarily been played yet.
        else:
            game["WINNER_NO"] = None
            game["WINNER"] = None

        games.append(game)


# ---------------------------------------------------------
# STEP 4: CREATE GAME-LEVEL DATAFRAME
# ---------------------------------------------------------

games_df = pd.DataFrame(games)


# ---------------------------------------------------------
# STEP 5: CHECK THAT BOTH ROWS BELONG TO SAME ROUND
# ---------------------------------------------------------

# If the pairing logic is correct, both teams in a game
# should have the same CURRENT ROUND value.

round_mismatches = []

for year, year_data in matchups.groupby("YEAR"):

    year_data = year_data.reset_index(drop=True)

    for i in range(0, len(year_data) - 1, 2):

        team1_round = year_data.iloc[i]["CURRENT ROUND"]
        team2_round = year_data.iloc[i + 1]["CURRENT ROUND"]

        if team1_round != team2_round:

            round_mismatches.append({
                "YEAR": year,
                "ROW_1": i,
                "ROW_2": i + 1,
                "ROUND_1": team1_round,
                "ROUND_2": team2_round
            })


# ---------------------------------------------------------
# STEP 6: SAVE DATA
# ---------------------------------------------------------

games_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ---------------------------------------------------------
# STEP 7: QUALITY CHECKS
# ---------------------------------------------------------

historical_games = games_df[
    games_df["YEAR"] <= 2025
]

future_games = games_df[
    games_df["YEAR"] == 2026
]


print("\n--- Game-Level Tournament Dataset ---")

print(
    f"Total games: "
    f"{len(games_df)}"
)

print(
    f"Historical games (through 2025): "
    f"{len(historical_games)}"
)

print(
    f"2026 rows/games: "
    f"{len(future_games)}"
)

print(
    f"Games with known winners: "
    f"{historical_games['WINNER'].notna().sum()}"
)

print(
    f"Round pairing mismatches: "
    f"{len(round_mismatches)}"
)


print("\nGames by Year:")

print(
    games_df
    .groupby("YEAR")
    .size()
)


print("\nSample 2025 Games:")

print(
    games_df[
        games_df["YEAR"] == 2025
    ]
    .head(6)
    .to_string(index=False)
)