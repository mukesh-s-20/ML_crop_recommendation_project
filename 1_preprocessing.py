"""
=========================================================
  AI-BASED CROP RECOMMENDATION SYSTEM
  Step 1: Data Preprocessing & Exploratory Data Analysis
=========================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import os
import warnings
warnings.filterwarnings("ignore")

# ─── Paths ──────────────────────────────────────────────────────────────────
DATA_PATH   = "data/Crop_Recommendation.xlsx"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 60)
print("   CROP RECOMMENDATION – DATA PREPROCESSING")
print("=" * 60)

# ─── 1. Load Dataset ─────────────────────────────────────────────────────────
df = pd.read_excel(DATA_PATH)
print(f"\n[1] Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")
print(df.head())

# ─── 2. Basic Info ───────────────────────────────────────────────────────────
print("\n[2] Column Data Types:")
print(df.dtypes)

print("\n[3] Statistical Summary:")
print(df.describe().T.round(3))

# ─── 3. Missing Values Check ─────────────────────────────────────────────────
print("\n[4] Missing Values:")
print(df.isnull().sum())

# No missing values – confirmed.

# ─── 4. Duplicate Check ──────────────────────────────────────────────────────
dupes = df.duplicated().sum()
print(f"\n[5] Duplicate rows found: {dupes}")
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)
print(f"    After removal: {df.shape[0]} rows")

# ─── 5. Outlier Detection & Capping (IQR method) ─────────────────────────────
print("\n[6] Outlier Treatment (IQR capping per feature):")
features = ["Nitrogen", "Phosphorus", "Potassium",
            "Temperature", "Humidity", "pH_Value", "Rainfall"]

for col in features:
    Q1  = df[col].quantile(0.25)
    Q3  = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = ((df[col] < lower) | (df[col] > upper)).sum()
    df[col] = df[col].clip(lower=lower, upper=upper)
    print(f"    {col:15s}: {outliers} outliers capped")

# ─── 6. Label Encoding ───────────────────────────────────────────────────────
le = LabelEncoder()
df["Crop_Label"] = le.fit_transform(df["Crop"])
class_names = le.classes_.tolist()
print(f"\n[7] Label Encoding → {len(class_names)} crops: {class_names}")

# ─── 7. Feature / Target Split ───────────────────────────────────────────────
X = df[features].values
y = df["Crop_Label"].values

# ─── 8. Normalization ────────────────────────────────────────────────────────
scaler_std = StandardScaler()
scaler_mm  = MinMaxScaler()

X_std = scaler_std.fit_transform(X)   # Standard Scaling
X_mm  = scaler_mm.fit_transform(X)    # Min-Max Scaling

print("\n[8] Normalization applied:")
print(f"    StandardScaler  – mean ≈ {X_std.mean():.4f}, std ≈ {X_std.std():.4f}")
print(f"    MinMaxScaler    – min  ≈ {X_mm.min():.4f},  max ≈ {X_mm.max():.4f}")

# ─── 9. Train / Test Split ───────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_std, y, test_size=0.25, random_state=42, stratify=y
)
print(f"\n[9] Train-Test Split (75/25, stratified):")
print(f"    Train: {X_train.shape} | Test: {X_test.shape}")

# ─── 10. Save Preprocessed Data ──────────────────────────────────────────────
np.save("data/X_train.npy", X_train)
np.save("data/X_test.npy",  X_test)
np.save("data/y_train.npy", y_train)
np.save("data/y_test.npy",  y_test)
np.save("data/X_raw.npy",   X_std)
np.save("data/y_raw.npy",   y)
np.save("data/X_mm.npy",    X_mm)

import joblib
joblib.dump(le,         "models/label_encoder.pkl")
joblib.dump(scaler_std, "models/scaler_std.pkl")
joblib.dump(class_names,"models/class_names.pkl")
print("\n    Preprocessed arrays & encoders saved.")

# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────

palette = sns.color_palette("Set2", len(class_names))

# Plot 1 – Class Distribution
plt.figure(figsize=(14, 5))
counts = pd.Series(y).map(lambda i: class_names[i]).value_counts().sort_values()
colors = sns.color_palette("husl", len(counts))
counts.plot(kind="barh", color=colors, edgecolor="white", linewidth=0.5)
plt.title("Crop Class Distribution in Dataset", fontsize=14, fontweight="bold")
plt.xlabel("Number of Samples")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/01_class_distribution.png", dpi=150)
plt.close()

# Plot 2 – Feature Distributions (before & after scaling)
fig, axes = plt.subplots(2, 7, figsize=(22, 7))
for i, col in enumerate(features):
    axes[0, i].hist(df[col], bins=30, color="#4C9BE8", edgecolor="white", linewidth=0.5)
    axes[0, i].set_title(col, fontsize=9, fontweight="bold")
    axes[0, i].set_ylabel("Freq" if i == 0 else "")
    axes[1, i].hist(X_std[:, i], bins=30, color="#F4845F", edgecolor="white", linewidth=0.5)
    axes[1, i].set_title(f"{col} (scaled)", fontsize=9)
    axes[1, i].set_ylabel("Freq" if i == 0 else "")

axes[0, 0].set_ylabel("Before Scaling", fontsize=10)
axes[1, 0].set_ylabel("After Scaling",  fontsize=10)
plt.suptitle("Feature Distributions: Before vs After StandardScaler", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/02_feature_distributions.png", dpi=150)
plt.close()

# Plot 3 – Correlation Heatmap
plt.figure(figsize=(9, 7))
corr = df[features].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.5, square=True, cbar_kws={"shrink": 0.8})
plt.title("Feature Correlation Matrix", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/03_correlation_heatmap.png", dpi=150)
plt.close()

# Plot 4 – Box Plots (after outlier capping)
fig, axes = plt.subplots(1, 7, figsize=(22, 5))
for i, col in enumerate(features):
    axes[i].boxplot(df[col], patch_artist=True,
                    boxprops=dict(facecolor=palette[i % len(palette)], color="gray"),
                    medianprops=dict(color="black", linewidth=2))
    axes[i].set_title(col, fontsize=9, fontweight="bold")
    axes[i].set_xticks([])
plt.suptitle("Box Plots After Outlier Capping", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/04_boxplots.png", dpi=150)
plt.close()

print(f"\n[10] All EDA plots saved to '{RESULTS_DIR}/'")
print("\n✅  Preprocessing complete!\n")
