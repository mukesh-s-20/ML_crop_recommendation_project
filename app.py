"""
=========================================================
  AI-BASED CROP RECOMMENDATION SYSTEM
  Step 6: Flask Web Application
=========================================================
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
import numpy as np
import joblib
import os
import json
import pandas as pd

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── Load Models & Encoders ──────────────────────────────────────────────────
scaler      = joblib.load(os.path.join(BASE_DIR, "models/scaler_4feat.pkl"))
le          = joblib.load(os.path.join(BASE_DIR, "models/label_encoder.pkl"))
class_names = joblib.load(os.path.join(BASE_DIR, "models/class_names.pkl"))
best_clf    = joblib.load(os.path.join(BASE_DIR, "models/best_classifier.pkl"))
mlp_model   = joblib.load(os.path.join(BASE_DIR, "models/mlp_model.pkl"))
best_info   = joblib.load(os.path.join(BASE_DIR, "models/best_model_info.pkl"))

# Load results
clf_results = pd.read_csv(os.path.join(BASE_DIR, "results/classification_results.csv"))
clust_results = pd.read_csv(os.path.join(BASE_DIR, "results/clustering_results.csv"))
dl_metrics  = joblib.load(os.path.join(BASE_DIR, "models/dl_metrics.pkl"))

# Crop info dictionary
CROP_INFO = {
    "Rice":        {"emoji": "🌾", "desc": "Requires high humidity and moderate temperature. Grows well in waterlogged conditions.", "season": "Kharif", "water": "High"},
    "Maize":       {"emoji": "🌽", "desc": "Needs warm weather and moderate rainfall. Suitable for well-drained soils.", "season": "Kharif/Rabi", "water": "Moderate"},
    "ChickPea":    {"emoji": "🫘", "desc": "A cool-season legume that grows well in semi-arid regions.", "season": "Rabi", "water": "Low"},
    "KidneyBeans": {"emoji": "🫘", "desc": "Requires warm temperatures and moderate moisture.", "season": "Kharif", "water": "Moderate"},
    "PigeonPeas":  {"emoji": "🌿", "desc": "Drought-tolerant legume. Thrives in warm tropical conditions.", "season": "Kharif", "water": "Low"},
    "MothBeans":   {"emoji": "🌱", "desc": "Extremely drought-resistant crop, suitable for arid regions.", "season": "Kharif", "water": "Very Low"},
    "MungBean":    {"emoji": "🫘", "desc": "Fast-growing legume, suitable for warm and humid conditions.", "season": "Kharif", "water": "Moderate"},
    "Blackgram":   {"emoji": "🫘", "desc": "Grows in tropical climate. Rich in protein and micronutrients.", "season": "Kharif/Rabi", "water": "Moderate"},
    "Lentil":      {"emoji": "🫘", "desc": "Cool-season crop. Fixes nitrogen in the soil.", "season": "Rabi", "water": "Low"},
    "Pomegranate": {"emoji": "🍎", "desc": "Drought-tolerant fruit. Grows well in semi-arid regions.", "season": "Perennial", "water": "Low"},
    "Banana":      {"emoji": "🍌", "desc": "Tropical fruit requiring high humidity and warm climate.", "season": "Perennial", "water": "High"},
    "Mango":       {"emoji": "🥭", "desc": "Tropical fruit tree needing well-drained soil and dry season.", "season": "Perennial", "water": "Low"},
    "Grapes":      {"emoji": "🍇", "desc": "Grows in temperate to subtropical climate. Needs dry summers.", "season": "Perennial", "water": "Low"},
    "Watermelon":  {"emoji": "🍉", "desc": "Requires warm weather and sandy loam soil with good drainage.", "season": "Kharif", "water": "Moderate"},
    "Muskmelon":   {"emoji": "🍈", "desc": "Warm season crop requiring well-drained sandy loam soil.", "season": "Kharif", "water": "Moderate"},
    "Apple":       {"emoji": "🍎", "desc": "Temperate fruit requiring cold winters and mild summers.", "season": "Perennial", "water": "Moderate"},
    "Orange":      {"emoji": "🍊", "desc": "Subtropical citrus fruit needing warm days and cool nights.", "season": "Perennial", "water": "Moderate"},
    "Papaya":      {"emoji": "🍈", "desc": "Fast-growing tropical fruit. Sensitive to frost.", "season": "Perennial", "water": "Moderate"},
    "Coconut":     {"emoji": "🥥", "desc": "Coastal tropical crop needing high humidity and warm climate.", "season": "Perennial", "water": "High"},
    "Cotton":      {"emoji": "☁️", "desc": "Needs long growing season, high temperature and moderate rainfall.", "season": "Kharif", "water": "Moderate"},
    "Jute":        {"emoji": "🌿", "desc": "Requires hot humid climate and well-distributed rainfall.", "season": "Kharif", "water": "High"},
    "Coffee":      {"emoji": "☕", "desc": "Grows best in tropical highlands with moderate temperature.", "season": "Perennial", "water": "Moderate"},
}

# ─── Routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict")
def predict_page():
    return render_template("predict.html", class_names=class_names)

@app.route("/dashboard")
def dashboard():
    clf_data     = clf_results.to_dict(orient="records")
    clust_data   = clust_results.to_dict(orient="records")
    return render_template("dashboard.html",
                           clf_data=clf_data,
                           clust_data=clust_data,
                           dl_metrics=dl_metrics,
                           best_model=best_info)

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data = request.get_json()
        features = np.array([[
            float(data["nitrogen"]),
            float(data["phosphorus"]),
            float(data["potassium"]),
            float(data["temperature"]),
        ]])

        features_scaled = scaler.transform(features)  # 4 features

        # Best classifier prediction
        pred_label   = best_clf.predict(features_scaled)[0]
        pred_proba   = best_clf.predict_proba(features_scaled)[0]
        crop_name    = class_names[pred_label]

        # MLP prediction (same 4 features)
        mlp_label    = mlp_model.predict(features_scaled)[0]
        mlp_proba    = mlp_model.predict_proba(features_scaled)[0]
        mlp_crop     = class_names[mlp_label]

        # Top 3 recommendations
        top3_idx     = np.argsort(pred_proba)[::-1][:3]
        top3         = [{"crop": class_names[i], "confidence": round(pred_proba[i] * 100, 1),
                         "emoji": CROP_INFO.get(class_names[i], {}).get("emoji", "🌱")}
                        for i in top3_idx]

        crop_info = CROP_INFO.get(crop_name, {
            "emoji": "🌱", "desc": "A suitable crop for your conditions.",
            "season": "Varies", "water": "Moderate"
        })

        return jsonify({
            "success":    True,
            "crop":       crop_name,
            "confidence": round(float(pred_proba[pred_label]) * 100, 1),
            "model":      best_info["name"],
            "mlp_crop":   mlp_crop,
            "mlp_conf":   round(float(mlp_proba[mlp_label]) * 100, 1),
            "top3":       top3,
            "info":       crop_info,
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route("/api/stats")
def api_stats():
    return jsonify({
        "classification": clf_results.to_dict(orient="records"),
        "clustering":     clust_results.to_dict(orient="records"),
        "deep_learning":  dl_metrics,
        "best_model":     best_info,
    })

@app.route("/results/<filename>")
def results_file(filename):
    return send_from_directory(os.path.join(BASE_DIR, "results"), filename)


@app.route("/api/crop_requirements/<crop_name>")
def api_crop_requirements(crop_name):
    import json, os
    req_path = os.path.join(BASE_DIR, "data/crop_soil_requirements.json")
    try:
        with open(req_path) as f:
            all_reqs = json.load(f)
        if crop_name not in all_reqs:
            return jsonify({"success": False, "error": f"Crop '{crop_name}' not found"}), 404
        return jsonify({
            "success": True,
            "crop": crop_name,
            "requirements": all_reqs[crop_name],
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/all_crops")
def api_all_crops():
    import json, os
    req_path = os.path.join(BASE_DIR, "data/crop_soil_requirements.json")
    with open(req_path) as f:
        all_reqs = json.load(f)
    return jsonify({"success": True, "crops": list(all_reqs.keys())})

if __name__ == "__main__":
    print("\n🌾  Crop Recommendation Web App")
    print("    URL: http://127.0.0.1:5000\n")
    app.run(debug=True, port=5000)
