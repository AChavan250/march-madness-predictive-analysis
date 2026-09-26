from pathlib import Path
import pandas as pd


# --- FILE PATHS ---
PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"

KENPOM_FILE = DATA_DIR / "KenPom Barttorvik.csv"
TEAMRANKINGS_FILE = DATA_DIR / "TeamRankings-selected-columns.csv"
RESUMES_FILE = DATA_DIR / "Resumes.csv"
AP_FILE = DATA_DIR / "AP Poll Data.csv"

OUTPUT_FILE = DATA_DIR / "team_features_base.csv"


# --- STEP 1: LOAD DATA ---
kenpom = pd.read_csv(KENPOM_FILE)
teamrankings = pd.read_csv(TEAMRANKINGS_FILE)
resumes = pd.read_csv(RESUMES_FILE)
ap = pd.read_csv(AP_FILE)


# --- STEP 2: SELECT BASE FEATURES ---
kenpom_features = kenpom[
    [
        "YEAR",
        "TEAM NO",
        "TEAM",
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
    ]
].copy()


teamrankings_features = teamrankings[
    [
        "YEAR",
        "TEAM NO",
        "TR RATING",
    ]
].copy()


resume_features = resumes[
    [
        "YEAR",
        "TEAM NO",
        "ELO",
        "B POWER",
        "Q1 PLUS Q2 W",
        "R SCORE",
    ]
].copy()


# --- STEP 3: GET LATEST AP POLL FOR EACH TEAM ---
ap_latest = (
    ap.sort_values(["YEAR", "TEAM NO", "WEEK"])
    .groupby(["YEAR", "TEAM NO"], as_index=False)
    .tail(1)
)

ap_features = ap_latest[
    [
        "YEAR",
        "TEAM NO",
        "AP RANK",
        "AP VOTES",
    ]
].copy()


# --- STEP 4: MERGE DATASETS ---
team_features = kenpom_features.merge(
    teamrankings_features,
    on=["YEAR", "TEAM NO"],
    how="left"
)

team_features = team_features.merge(
    resume_features,
    on=["YEAR", "TEAM NO"],
    how="left"
)

team_features = team_features.merge(
    ap_features,
    on=["YEAR", "TEAM NO"],
    how="left"
)


# --- STEP 5: SORT AND SAVE ---
team_features = team_features.sort_values(
    ["YEAR", "TEAM"]
).reset_index(drop=True)

team_features.to_csv(
    OUTPUT_FILE,
    index=False
)


# --- STEP 6: QUICK QUALITY CHECK ---
print("\n--- Base Team Feature Dataset ---")
print(f"Rows: {len(team_features)}")
print(f"Columns: {len(team_features.columns)}")

print("\nYears:")
print(
    sorted(
        team_features["YEAR"]
        .dropna()
        .unique()
    )
)

print("\nMissing Values:")
print(
    team_features
    .isna()
    .sum()
    .sort_values(ascending=False)
)

print("\nSample:")
print(team_features.head())