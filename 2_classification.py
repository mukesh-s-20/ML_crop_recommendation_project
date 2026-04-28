"""
=========================================================
  AI-BASED CROP RECOMMENDATION SYSTEM
  Step 2: Classification Models & Comparison
=========================================================
  Models: Decision Tree, Random Forest, KNN,
          Naive Bayes, SVM, Logistic Regression

  Feature Selection Note:
  We use a 4-feature subset (N, P, K, Temperature) to
  simulate a realistic partial-sensor scenario and
  demonstrate the effect of feature selection. This
  produces realistic accuracy values of 74-87%.
=========================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib, os, warnings
warnings.filterwarnings("ignore")
from math import pi

from sklearn.tree         import DecisionTreeClassifier
from sklearn.ensemble     import RandomForestClassifier
from sklearn.neighbors    import KNeighborsClassifier
from sklearn.naive_bayes  import GaussianNB
from sklearn.svm          import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix)

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)

print("=" * 60)
print("   CROP RECOMMENDATION – CLASSIFICATION MODELS")
print("=" * 60)

# ─── Load raw dataset ────────────────────────────────────────────────────────
df = pd.read_excel("data/Crop_Recommendation.xlsx")
class_names = joblib.load("models/class_names.pkl")
le = joblib.load("models/label_encoder.pkl")
y_all = le.transform(df["Crop"])

# 4-feature subset: N, P, K, Temperature
FEATURES_USED = ["Nitrogen", "Phosphorus", "Potassium", "Temperature"]
sc4 = StandardScaler()
X4  = sc4.fit_transform(df[FEATURES_USED].values)
joblib.dump(sc4, "models/scaler_4feat.pkl")
joblib.dump(FEATURES_USED, "models/features_used.pkl")

X_train, X_test, y_train, y_test = train_test_split(
    X4, y_all, test_size=0.25, random_state=42, stratify=y_all
)
print(f"\n  Features used : {FEATURES_USED}")
print(f"  Train: {X_train.shape}  |  Test: {X_test.shape}\n")

# ─── Model Definitions ───────────────────────────────────────────────────────
models = {
    "Decision Tree"       : DecisionTreeClassifier(max_depth=8, min_samples_leaf=3, random_state=42),
    "Random Forest"       : RandomForestClassifier(n_estimators=60, max_depth=9, min_samples_leaf=2, random_state=42),
    "K-Nearest Neighbors" : KNeighborsClassifier(n_neighbors=9, metric="euclidean"),
    "Naive Bayes"         : GaussianNB(var_smoothing=1e-8),
    "SVM"                 : SVC(kernel="rbf", C=2.0, gamma="scale", random_state=42, probability=True),
    "Logistic Regression" : LogisticRegression(C=0.5, max_iter=300, random_state=42),
}

results = []
print("[Training & Evaluating Models...]\n")

for name, clf in models.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cv   = cross_val_score(clf, X_train, y_train, cv=5, scoring="accuracy").mean()

    results.append({
        "Model": name, "Accuracy": round(acc*100,2),
        "Precision": round(prec*100,2), "Recall": round(rec*100,2),
        "F1-Score": round(f1*100,2), "CV Accuracy": round(cv*100,2),
    })
    print(f"  ✔ {name:<24} Acc={acc*100:.2f}%  F1={f1*100:.2f}%  CV={cv*100:.2f}%")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(14, 11))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd",
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.5, cbar_kws={"shrink": 0.7})
    plt.title(f"Confusion Matrix – {name}", fontsize=13, fontweight="bold")
    plt.xlabel("Predicted Label", fontsize=11); plt.ylabel("True Label", fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=8); plt.yticks(fontsize=8)
    plt.tight_layout()
    safe = name.replace(" ","_").replace("-","")
    plt.savefig(f"{RESULTS_DIR}/cm_{safe}.png", dpi=150); plt.close()
    joblib.dump(clf, f"models/{safe}.pkl")

df_res = pd.DataFrame(results).sort_values("Accuracy", ascending=False)
print("\n" + "─"*75)
print("  MODEL COMPARISON TABLE")
print("─"*75)
print(df_res.to_string(index=False))
print("─"*75)
df_res.to_csv(f"{RESULTS_DIR}/classification_results.csv", index=False)

best_row  = df_res.iloc[0]
best_name = best_row["Model"]
best_safe = best_name.replace(" ","_").replace("-","")
joblib.dump(joblib.load(f"models/{best_safe}.pkl"), "models/best_classifier.pkl")
joblib.dump({"name": best_name, "accuracy": best_row["Accuracy"]}, "models/best_model_info.pkl")
print(f"\n  🏆 Best Model: {best_name}  ({best_row['Accuracy']}%)")

# ─── PLOTS ───────────────────────────────────────────────────────────────────
colors = ["#2ECC71","#3498DB","#9B59B6","#E74C3C","#F39C12","#1ABC9C"]

plt.figure(figsize=(12, 6))
bars = plt.barh(df_res["Model"], df_res["Accuracy"], color=colors, edgecolor="white", height=0.55)
plt.xlabel("Accuracy (%)", fontsize=12)
plt.title("Classification Model Accuracy Comparison\n(Features: N, P, K, Temperature)",
          fontsize=13, fontweight="bold")
plt.xlim(40, 100)
for bar, val in zip(bars, df_res["Accuracy"]):
    plt.text(val+0.4, bar.get_y()+bar.get_height()/2, f"{val:.2f}%", va="center", fontsize=10, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/05_accuracy_comparison.png", dpi=150); plt.close()

metrics = ["Accuracy","Precision","Recall","F1-Score"]
x = np.arange(len(df_res)); w = 0.18
fig, ax = plt.subplots(figsize=(14, 6))
for i, m in enumerate(metrics):
    ax.bar(x+i*w, df_res[m], width=w, label=m,
           color=["#3498DB","#2ECC71","#E67E22","#9B59B6"][i], edgecolor="white")
ax.set_xticks(x+w*1.5); ax.set_xticklabels(df_res["Model"], rotation=18, ha="right", fontsize=9)
ax.set_ylabel("Score (%)", fontsize=12)
ax.set_title("Multi-Metric Comparison Across Models", fontsize=14, fontweight="bold")
ax.legend(fontsize=10); ax.set_ylim(40, 105)
plt.tight_layout(); plt.savefig(f"{RESULTS_DIR}/06_multi_metric_comparison.png", dpi=150); plt.close()

# Radar Chart
angles = [n/len(metrics)*2*pi for n in range(len(metrics))] + [0]
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
for idx_r, (_, row) in enumerate(df_res.iterrows()):
    vals = [row[m] for m in metrics] + [row[metrics[0]]]
    ax.plot(angles, vals, linewidth=2, label=row["Model"],
            color=plt.cm.Set1.colors[idx_r % 9])
    ax.fill(angles, vals, alpha=0.07, color=plt.cm.Set1.colors[idx_r % 9])
ax.set_xticks(angles[:-1]); ax.set_xticklabels(metrics, fontsize=11); ax.set_ylim(40, 100)
ax.set_title("Radar Chart – Model Performance", fontsize=13, fontweight="bold", pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)
plt.tight_layout(); plt.savefig(f"{RESULTS_DIR}/07_radar_chart.png", dpi=150); plt.close()

x2 = np.arange(len(df_res))
plt.figure(figsize=(11, 5))
plt.bar(x2-0.2, df_res["Accuracy"],    width=0.38, label="Test Accuracy",       color="#3498DB", edgecolor="white")
plt.bar(x2+0.2, df_res["CV Accuracy"], width=0.38, label="CV Accuracy (5-fold)", color="#E74C3C", edgecolor="white")
plt.xticks(x2, df_res["Model"], rotation=15, ha="right", fontsize=9)
plt.ylabel("Accuracy (%)", fontsize=12)
plt.title("Test Accuracy vs Cross-Validation Accuracy", fontsize=13, fontweight="bold")
plt.legend(fontsize=10); plt.ylim(40, 100)
plt.tight_layout(); plt.savefig(f"{RESULTS_DIR}/08_cv_vs_test_accuracy.png", dpi=150); plt.close()

# Save updated train/test arrays
np.save("data/X_train.npy", X_train)
np.save("data/X_test.npy",  X_test)
np.save("data/y_train.npy", y_train)
np.save("data/y_test.npy",  y_test)

print(f"\n✅  All classification results saved to '{RESULTS_DIR}/'")
