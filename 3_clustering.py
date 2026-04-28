"""
=========================================================
  AI-BASED CROP RECOMMENDATION SYSTEM
  Step 3: Unsupervised Learning – Clustering
=========================================================
  Algorithms: K-Means, DBSCAN, Hierarchical (Agglomerative)
              + Apriori (association rule-based grouping)
=========================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.cluster         import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition   import PCA
from sklearn.metrics         import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 60)
print("   CROP RECOMMENDATION – UNSUPERVISED CLUSTERING")
print("=" * 60)

# ─── Load Data ───────────────────────────────────────────────────────────────
X = np.load("data/X_raw.npy")     # StandardScaled full dataset
y = np.load("data/y_raw.npy")
class_names = joblib.load("models/class_names.pkl")
N_CLUSTERS  = 22   # equals number of crop types

# ─── PCA for Visualization ───────────────────────────────────────────────────
pca2 = PCA(n_components=2, random_state=42)
X2   = pca2.fit_transform(X)

pca3 = PCA(n_components=3, random_state=42)
X3   = pca3.fit_transform(X)

palette22 = plt.cm.tab20.colors + plt.cm.Set3.colors[:2]

def plot_clusters(X2d, labels, title, filename, true_labels=None):
    fig, axes = plt.subplots(1, 2 if true_labels is not None else 1,
                             figsize=(14 if true_labels is not None else 8, 6))
    if true_labels is None:
        axes = [axes]

    unique = np.unique(labels[labels != -1])
    for cl in unique:
        mask = labels == cl
        col  = palette22[int(cl) % len(palette22)] if cl != -1 else "black"
        axes[0].scatter(X2d[mask, 0], X2d[mask, 1], s=18, alpha=0.6,
                        color=col, label=f"C{cl}")
    if np.any(labels == -1):
        axes[0].scatter(X2d[labels == -1, 0], X2d[labels == -1, 1],
                        s=18, alpha=0.4, color="lightgray", label="Noise")
    axes[0].set_title(f"{title}\n(Predicted Clusters)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("PCA Component 1")
    axes[0].set_ylabel("PCA Component 2")

    if true_labels is not None:
        for cl in np.unique(true_labels):
            mask = true_labels == cl
            axes[1].scatter(X2d[mask, 0], X2d[mask, 1], s=18, alpha=0.6,
                            color=palette22[int(cl) % len(palette22)],
                            label=class_names[int(cl)])
        axes[1].set_title("True Crop Labels (PCA)", fontsize=11, fontweight="bold")
        axes[1].set_xlabel("PCA Component 1")
        axes[1].set_ylabel("PCA Component 2")

    plt.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/{filename}", dpi=150)
    plt.close()

cluster_results = []

# ─────────────────────────────────────────────────────────────────────────────
# 1. K-MEANS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1] K-Means Clustering ...")

# Elbow curve
inertias = []
k_range  = range(2, 31)
for k in k_range:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    km.fit(X)
    inertias.append(km.inertia_)

plt.figure(figsize=(9, 5))
plt.plot(k_range, inertias, "b-o", markersize=5)
plt.axvline(N_CLUSTERS, color="red", linestyle="--", label=f"k={N_CLUSTERS}")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia (WCSS)")
plt.title("K-Means Elbow Curve", fontsize=13, fontweight="bold")
plt.legend()
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/09_kmeans_elbow.png", dpi=150)
plt.close()

km_final = KMeans(n_clusters=N_CLUSTERS, n_init=15, random_state=42)
km_labels = km_final.fit_predict(X)
joblib.dump(km_final, "models/kmeans.pkl")

sil  = silhouette_score(X, km_labels)
db   = davies_bouldin_score(X, km_labels)
ch   = calinski_harabasz_score(X, km_labels)
print(f"   Silhouette={sil:.4f}  Davies-Bouldin={db:.4f}  Calinski-Harabasz={ch:.2f}")

cluster_results.append({"Algorithm": "K-Means", "Silhouette": round(sil, 4),
                         "Davies-Bouldin": round(db, 4), "Calinski-Harabasz": round(ch, 2),
                         "Clusters Found": N_CLUSTERS})

plot_clusters(X2, km_labels, "K-Means Clustering (k=22)", "10_kmeans_clusters.png", y)

# ─────────────────────────────────────────────────────────────────────────────
# 2. DBSCAN
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2] DBSCAN Clustering ...")

db_model  = DBSCAN(eps=0.65, min_samples=5)
db_labels = db_model.fit_predict(X)
joblib.dump(db_model, "models/dbscan.pkl")

n_clusters_db = len(set(db_labels)) - (1 if -1 in db_labels else 0)
n_noise       = np.sum(db_labels == -1)
print(f"   Clusters found: {n_clusters_db}  |  Noise points: {n_noise}")

valid = db_labels != -1
if valid.sum() > 1 and len(np.unique(db_labels[valid])) > 1:
    sil_db = silhouette_score(X[valid], db_labels[valid])
    db_db  = davies_bouldin_score(X[valid], db_labels[valid])
    ch_db  = calinski_harabasz_score(X[valid], db_labels[valid])
else:
    sil_db, db_db, ch_db = 0.0, 0.0, 0.0

print(f"   Silhouette={sil_db:.4f}  Davies-Bouldin={db_db:.4f}  Calinski-Harabasz={ch_db:.2f}")
cluster_results.append({"Algorithm": "DBSCAN", "Silhouette": round(sil_db, 4),
                         "Davies-Bouldin": round(db_db, 4), "Calinski-Harabasz": round(ch_db, 2),
                         "Clusters Found": n_clusters_db})

plot_clusters(X2, db_labels, "DBSCAN Clustering", "11_dbscan_clusters.png", y)

# ─────────────────────────────────────────────────────────────────────────────
# 3. HIERARCHICAL (Agglomerative)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3] Hierarchical (Agglomerative) Clustering ...")

hc_model  = AgglomerativeClustering(n_clusters=N_CLUSTERS, linkage="ward")
hc_labels = hc_model.fit_predict(X)
joblib.dump(hc_model, "models/hierarchical.pkl")

sil_hc = silhouette_score(X, hc_labels)
db_hc  = davies_bouldin_score(X, hc_labels)
ch_hc  = calinski_harabasz_score(X, hc_labels)
print(f"   Silhouette={sil_hc:.4f}  Davies-Bouldin={db_hc:.4f}  Calinski-Harabasz={ch_hc:.2f}")

cluster_results.append({"Algorithm": "Hierarchical", "Silhouette": round(sil_hc, 4),
                         "Davies-Bouldin": round(db_hc, 4), "Calinski-Harabasz": round(ch_hc, 2),
                         "Clusters Found": N_CLUSTERS})

plot_clusters(X2, hc_labels, "Hierarchical Clustering (Ward Linkage)", "12_hierarchical_clusters.png", y)

# Dendrogram (on 10% sample for speed)
sample_idx = np.random.choice(len(X), size=300, replace=False)
linked = linkage(X[sample_idx], method="ward")
plt.figure(figsize=(16, 6))
dendrogram(linked, truncate_mode="level", p=5,
           leaf_font_size=8, color_threshold=0.7 * max(linked[:, 2]))
plt.title("Hierarchical Clustering Dendrogram (300-sample)", fontsize=13, fontweight="bold")
plt.xlabel("Sample Index")
plt.ylabel("Euclidean Distance")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/13_dendrogram.png", dpi=150)
plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# 4. APRIORI-STYLE CLUSTERING (Frequency-based Binning + Association)
#    (Implemented from scratch – no mlxtend dependency)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4] Apriori-Style Cluster Analysis ...")

# We discretise each feature into Low/Medium/High bins, then assign
# each sample to its cluster based on which 'pattern' it belongs to.
df_bin = pd.DataFrame(
    np.load("data/X_raw.npy"),
    columns=["Nitrogen", "Phosphorus", "Potassium",
             "Temperature", "Humidity", "pH_Value", "Rainfall"]
)
df_bin["Crop"] = [class_names[i] for i in np.load("data/y_raw.npy")]

# Bin
for col in ["Nitrogen", "Phosphorus", "Potassium",
            "Temperature", "Humidity", "pH_Value", "Rainfall"]:
    df_bin[f"{col}_bin"] = pd.qcut(df_bin[col], q=3, labels=["Low", "Med", "High"])

bin_cols  = [c for c in df_bin.columns if c.endswith("_bin")]
df_bin["pattern"] = df_bin[bin_cols].astype(str).agg("|".join, axis=1)

pattern_counts = df_bin["pattern"].value_counts()
apriori_labels = df_bin["pattern"].map(
    {p: i for i, p in enumerate(pattern_counts.index)}
).values.astype(int)

# Reduce to N_CLUSTERS representative clusters
top_patterns = pattern_counts.index[:N_CLUSTERS]
apriori_labels_capped = np.where(
    df_bin["pattern"].isin(top_patterns),
    df_bin["pattern"].map({p: i for i, p in enumerate(top_patterns)}),
    N_CLUSTERS  # "other" bin
).astype(int)

sil_ap = silhouette_score(X, apriori_labels_capped)
db_ap  = davies_bouldin_score(X, apriori_labels_capped)
ch_ap  = calinski_harabasz_score(X, apriori_labels_capped)
print(f"   Patterns found: {len(pattern_counts)}  |  Top clusters: {N_CLUSTERS}")
print(f"   Silhouette={sil_ap:.4f}  Davies-Bouldin={db_ap:.4f}  Calinski-Harabasz={ch_ap:.2f}")

cluster_results.append({"Algorithm": "Apriori-Cluster", "Silhouette": round(sil_ap, 4),
                         "Davies-Bouldin": round(db_ap, 4), "Calinski-Harabasz": round(ch_ap, 2),
                         "Clusters Found": N_CLUSTERS})

plot_clusters(X2, apriori_labels_capped, "Apriori-Style Cluster Assignment", "14_apriori_clusters.png", y)

# ─────────────────────────────────────────────────────────────────────────────
# COMPARISON TABLE & PLOTS
# ─────────────────────────────────────────────────────────────────────────────
df_cl = pd.DataFrame(cluster_results)
print("\n\n" + "─" * 70)
print("  CLUSTERING COMPARISON TABLE")
print("─" * 70)
print(df_cl.to_string(index=False))
print("─" * 70)

df_cl.to_csv(f"{RESULTS_DIR}/clustering_results.csv", index=False)

# Plot – Silhouette Scores
plt.figure(figsize=(9, 5))
colors = ["#3498DB", "#E74C3C", "#2ECC71", "#F39C12"]
plt.bar(df_cl["Algorithm"], df_cl["Silhouette"], color=colors, edgecolor="white", width=0.5)
for i, v in enumerate(df_cl["Silhouette"]):
    plt.text(i, v + 0.005, f"{v:.4f}", ha="center", fontsize=10, fontweight="bold")
plt.ylabel("Silhouette Score (higher = better)", fontsize=11)
plt.title("Clustering Algorithm Comparison – Silhouette Score", fontsize=13, fontweight="bold")
plt.ylim(0, 0.5)
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/15_clustering_comparison.png", dpi=150)
plt.close()

# Plot – Davies-Bouldin (lower = better)
plt.figure(figsize=(9, 5))
plt.bar(df_cl["Algorithm"], df_cl["Davies-Bouldin"], color=colors, edgecolor="white", width=0.5)
for i, v in enumerate(df_cl["Davies-Bouldin"]):
    plt.text(i, v + 0.02, f"{v:.4f}", ha="center", fontsize=10, fontweight="bold")
plt.ylabel("Davies-Bouldin Index (lower = better)", fontsize=11)
plt.title("Clustering Algorithm Comparison – Davies-Bouldin Index", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/16_davies_bouldin.png", dpi=150)
plt.close()

print(f"\n✅  All clustering results & plots saved to '{RESULTS_DIR}/'")
