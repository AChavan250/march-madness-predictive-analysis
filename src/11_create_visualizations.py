from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_DIR / "outputs"


# =========================================================
# 1. MODEL COMPARISON
# =========================================================

models = pd.read_csv(
    OUTPUT_DIR / "model_comparison.csv"
)

models = models.sort_values(
    "VALIDATION_ACCURACY",
    ascending=True
)

plt.figure(figsize=(10, 6))

y = range(len(models))

plt.barh(
    y,
    models["VALIDATION_ACCURACY"] * 100,
    height=0.35,
    label="Validation 2021–2024"
)

plt.barh(
    [i + 0.35 for i in y],
    models["TEST_2025_ACCURACY"] * 100,
    height=0.35,
    label="2025 Test"
)

plt.yticks(
    [i + 0.175 for i in y],
    models["MODEL"]
)

plt.xlabel("Accuracy (%)")
plt.title("March Madness Model Accuracy")
plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "model_accuracy_comparison.png",
    dpi=300
)

plt.close()


# =========================================================
# 2. BETTING ROI
# =========================================================

roi = pd.read_csv(
    OUTPUT_DIR / "betting_roi_results.csv"
)

roi = roi.sort_values(
    "ROI",
    ascending=True
)

plt.figure(figsize=(9, 5))

plt.barh(
    roi["MODEL"],
    roi["ROI"]
)

plt.axvline(
    0,
    linewidth=1
)

plt.xlabel("Estimated ROI (%)")
plt.title(
    "Model vs Vegas: Hypothetical Underdog Betting ROI"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "betting_roi_comparison.png",
    dpi=300
)

plt.close()


# =========================================================
# 3. RANDOM FOREST FEATURE IMPORTANCE
# =========================================================

importance = pd.read_csv(
    OUTPUT_DIR / "random_forest_feature_importance.csv"
)

top10 = (
    importance
    .head(10)
    .sort_values(
        "IMPORTANCE",
        ascending=True
    )
)

plt.figure(figsize=(9, 6))

plt.barh(
    top10["FEATURE"],
    top10["IMPORTANCE"]
)

plt.xlabel("Feature Importance")
plt.title(
    "Top 10 Random Forest Predictors"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "random_forest_feature_importance.png",
    dpi=300
)

plt.close()


print("\nVisualizations created:")
print("outputs/model_accuracy_comparison.png")
print("outputs/betting_roi_comparison.png")
print("outputs/random_forest_feature_importance.png")