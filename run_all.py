"""
=========================================================
  AI-BASED CROP RECOMMENDATION SYSTEM
  run_all.py  –  Master Runner Script
=========================================================
  Run this file ONCE to:
    1. Preprocess data
    2. Train all classification models
    3. Run clustering algorithms
    4. Train deep learning model
    5. Mine association rules
    6. Launch the web application

  Usage:
      python run_all.py
=========================================================
"""

import subprocess
import sys
import os
import time

STEPS = [
    ("1_preprocessing.py",    "Data Preprocessing & EDA"),
    ("2_classification.py",   "Classification Models"),
    ("3_clustering.py",       "Unsupervised Clustering"),
    ("4_deep_learning.py",    "Deep Learning (MLP)"),
    ("5_association_rules.py","Association Rule Mining"),
]

def banner(text):
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)

def run_step(script, label):
    banner(f"STEP: {label}")
    t0  = time.time()
    ret = subprocess.run([sys.executable, script], cwd=os.path.dirname(os.path.abspath(__file__)))
    elapsed = time.time() - t0
    if ret.returncode != 0:
        print(f"\n❌  {label} FAILED (exit code {ret.returncode})")
        sys.exit(1)
    print(f"\n✅  {label} done ({elapsed:.1f}s)")

if __name__ == "__main__":
    banner("AI-BASED CROP RECOMMENDATION SYSTEM – FULL PIPELINE")
    print("  This will train all models and generate all plots.")
    print("  Estimated time: 2–5 minutes\n")

    total_start = time.time()

    for script, label in STEPS:
        run_step(script, label)

    total_time = time.time() - total_start
    banner(f"ALL STEPS COMPLETED in {total_time:.1f}s")

    print("\n  🌾  Starting Web Application ...")
    print("  Open your browser at: http://127.0.0.1:5000\n")
    print("  Press Ctrl+C to stop the server.\n")

    try:
        subprocess.run([sys.executable, "app.py"],
                       cwd=os.path.dirname(os.path.abspath(__file__)))
    except KeyboardInterrupt:
        print("\n\n  Server stopped. Goodbye! 🌾\n")
