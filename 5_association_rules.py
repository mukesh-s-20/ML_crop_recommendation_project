"""
=========================================================
  AI-BASED CROP RECOMMENDATION SYSTEM
  Step 5: Association Rule Mining (Apriori Algorithm)
=========================================================
  Implemented from scratch – no external dependency
  Finds frequent patterns among crop-growing conditions
=========================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import itertools
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 60)
print("   CROP RECOMMENDATION – ASSOCIATION RULE MINING (APRIORI)")
print("=" * 60)

# ─── Load & Bin Data ─────────────────────────────────────────────────────────
X_raw      = np.load("data/X_raw.npy")
y_raw      = np.load("data/y_raw.npy")
class_names = joblib.load("models/class_names.pkl")

features = ["Nitrogen", "Phosphorus", "Potassium",
            "Temperature", "Humidity", "pH_Value", "Rainfall"]

df = pd.DataFrame(X_raw, columns=features)
df["Crop"] = [class_names[i] for i in y_raw]

# Discretise each feature into 3 bins
for col in features:
    q33 = df[col].quantile(0.33)
    q66 = df[col].quantile(0.66)
    df[f"{col}_lvl"] = df[col].apply(
        lambda v: f"{col}_Low" if v <= q33 else (f"{col}_Med" if v <= q66 else f"{col}_High")
    )

lvl_cols = [c for c in df.columns if c.endswith("_lvl")]

# Create transactions: each row = set of items (feature levels + crop)
transactions = []
for _, row in df.iterrows():
    items = set([row[c] for c in lvl_cols] + [f"CROP_{row['Crop']}"])
    transactions.append(frozenset(items))

N = len(transactions)
print(f"\n  Total transactions : {N}")
print(f"  Unique items       : {len(set(i for t in transactions for i in t))}")

# ─── Apriori From Scratch ────────────────────────────────────────────────────
def get_support(transactions, itemset):
    return sum(1 for t in transactions if itemset.issubset(t)) / len(transactions)

def apriori(transactions, min_support=0.04, max_length=3):
    """Return frequent itemsets with support >= min_support."""
    # C1 – all single items
    all_items = set(i for t in transactions for i in t)
    freq_sets = {}

    # L1
    L1 = {}
    for item in all_items:
        sup = get_support(transactions, frozenset([item]))
        if sup >= min_support:
            L1[frozenset([item])] = sup
    freq_sets.update(L1)
    print(f"  L1 frequent items : {len(L1)}")

    Lk = L1
    k  = 2
    while Lk and k <= max_length:
        # Candidate generation (Apriori join)
        Lk_list = list(Lk.keys())
        Ck = set()
        for i in range(len(Lk_list)):
            for j in range(i + 1, len(Lk_list)):
                union = Lk_list[i] | Lk_list[j]
                if len(union) == k:
                    Ck.add(union)

        Lk_new = {}
        for cand in Ck:
            sup = get_support(transactions, cand)
            if sup >= min_support:
                Lk_new[cand] = sup

        print(f"  L{k} frequent sets  : {len(Lk_new)}")
        freq_sets.update(Lk_new)
        Lk = Lk_new
        k += 1

    return freq_sets

print("\n[Running Apriori (min_support=0.04, max_length=3)...]")
freq_sets = apriori(transactions, min_support=0.04, max_length=3)
print(f"\n  Total frequent itemsets: {len(freq_sets)}")

# ─── Association Rule Generation ─────────────────────────────────────────────
def generate_rules(freq_sets, transactions, min_confidence=0.50, min_lift=1.0):
    rules = []
    for itemset, sup in freq_sets.items():
        if len(itemset) < 2:
            continue
        for n in range(1, len(itemset)):
            for antecedent in itertools.combinations(itemset, n):
                antecedent  = frozenset(antecedent)
                consequent  = itemset - antecedent
                if not consequent:
                    continue
                sup_ant = get_support(transactions, antecedent)
                if sup_ant == 0:
                    continue
                conf = sup / sup_ant
                sup_cons = get_support(transactions, consequent)
                lift = conf / sup_cons if sup_cons > 0 else 0
                if conf >= min_confidence and lift >= min_lift:
                    rules.append({
                        "Antecedent" : ", ".join(sorted(antecedent)),
                        "Consequent" : ", ".join(sorted(consequent)),
                        "Support"    : round(sup, 4),
                        "Confidence" : round(conf, 4),
                        "Lift"       : round(lift, 4),
                    })
    return pd.DataFrame(rules).sort_values("Lift", ascending=False).reset_index(drop=True)

print("\n[Generating Association Rules (conf≥0.55, lift≥1.0)...]")
rules_df = generate_rules(freq_sets, transactions, min_confidence=0.50, min_lift=1.0)
print(f"  Total rules generated: {len(rules_df)}")

# ─── Filter Crop-related Rules ───────────────────────────────────────────────
crop_rules = rules_df[
    rules_df["Antecedent"].str.contains("CROP_") |
    rules_df["Consequent"].str.contains("CROP_")
].head(50)

print(f"  Crop-related rules  : {len(crop_rules)}")

# Top 20 rules display
top_rules = rules_df.head(20)
print("\n  Top 20 Rules by Lift:")
print(top_rules[["Antecedent","Consequent","Support","Confidence","Lift"]].to_string(index=False))

# Save
rules_df.to_csv(f"{RESULTS_DIR}/association_rules.csv", index=False)
crop_rules.to_csv(f"{RESULTS_DIR}/crop_association_rules.csv", index=False)
joblib.dump(rules_df, "models/association_rules.pkl")

# ─── Performance Metrics ─────────────────────────────────────────────────────
print("\n" + "─" * 55)
print("  ASSOCIATION RULE MINING – PERFORMANCE METRICS")
print("─" * 55)
print(f"  Total frequent itemsets   : {len(freq_sets)}")
print(f"  Total association rules   : {len(rules_df)}")
print(f"  Crop-specific rules       : {len(crop_rules)}")
if len(rules_df):
    print(f"  Avg Support    (all rules): {rules_df['Support'].mean():.4f}")
    print(f"  Avg Confidence (all rules): {rules_df['Confidence'].mean():.4f}")
    print(f"  Avg Lift       (all rules): {rules_df['Lift'].mean():.4f}")
    print(f"  Max Lift                  : {rules_df['Lift'].max():.4f}")
print("─" * 55)

# ─── PLOTS ───────────────────────────────────────────────────────────────────

if len(rules_df) > 5:
    # Plot 1 – Support vs Confidence scatter coloured by Lift
    plt.figure(figsize=(10, 6))
    sc = plt.scatter(rules_df["Support"], rules_df["Confidence"],
                     c=rules_df["Lift"], cmap="YlOrRd", alpha=0.7, s=40)
    plt.colorbar(sc, label="Lift")
    plt.xlabel("Support", fontsize=12)
    plt.ylabel("Confidence", fontsize=12)
    plt.title("Association Rules – Support vs Confidence (coloured by Lift)",
              fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/21_arm_scatter.png", dpi=150)
    plt.close()

    # Plot 2 – Top 15 Rules by Lift
    top15 = rules_df.head(15).copy()
    top15["Rule"] = top15.apply(
        lambda r: f"{r['Antecedent'][:30]}… → {r['Consequent'][:20]}", axis=1
    )
    plt.figure(figsize=(12, 7))
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, 15))
    plt.barh(top15["Rule"][::-1], top15["Lift"][::-1], color=colors, edgecolor="white")
    plt.xlabel("Lift", fontsize=12)
    plt.title("Top 15 Association Rules by Lift", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/22_arm_top_rules_lift.png", dpi=150)
    plt.close()

    # Plot 3 – Confidence distribution
    plt.figure(figsize=(9, 5))
    plt.hist(rules_df["Confidence"], bins=20, color="#3498DB", edgecolor="white")
    plt.axvline(rules_df["Confidence"].mean(), color="red", linestyle="--",
                label=f"Mean={rules_df['Confidence'].mean():.2f}")
    plt.xlabel("Confidence", fontsize=12)
    plt.ylabel("Number of Rules", fontsize=12)
    plt.title("Distribution of Rule Confidence Values", fontsize=13, fontweight="bold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/23_arm_confidence_dist.png", dpi=150)
    plt.close()

    # Plot 4 – Support vs Lift for crop rules
    if len(crop_rules) > 2:
        plt.figure(figsize=(9, 5))
        plt.scatter(crop_rules["Support"], crop_rules["Lift"],
                    c="#E74C3C", alpha=0.8, s=60, edgecolors="white")
        for _, row in crop_rules.head(8).iterrows():
            plt.annotate(row["Consequent"][:20], (row["Support"], row["Lift"]),
                         fontsize=7, alpha=0.8)
        plt.xlabel("Support", fontsize=12)
        plt.ylabel("Lift", fontsize=12)
        plt.title("Crop-Related Rules – Support vs Lift", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"{RESULTS_DIR}/24_arm_crop_rules.png", dpi=150)
        plt.close()

print(f"\n✅  Association Rule Mining complete – results saved to '{RESULTS_DIR}/'")
