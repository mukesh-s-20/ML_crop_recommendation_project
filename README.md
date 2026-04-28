# 🌾 AgroMind – AI Based Crop Recommendation System

A complete Machine Learning project for crop recommendation built as a Final Year / Mini Project.

---

## 📁 Project Structure

```
crop_recommendation_project/
├── data/
│   ├── Crop_Recommendation.xlsx     ← Original dataset
│   ├── X_train.npy / X_test.npy    ← Preprocessed arrays (4-feature)
│   └── y_train.npy / y_test.npy    ← Label arrays
│
├── models/                          ← Saved trained models (.pkl)
│   ├── best_classifier.pkl
│   ├── mlp_model.pkl
│   ├── kmeans.pkl / dbscan.pkl / hierarchical.pkl
│   ├── label_encoder.pkl
│   ├── scaler_4feat.pkl / scaler_std.pkl
│   └── class_names.pkl
│
├── results/                         ← All generated plots (PNG)
│
├── static/
│   ├── css/style.css
│   └── js/main.js, predict.js
│
├── templates/
│   ├── base.html
│   ├── index.html          ← Home page
│   ├── predict.html        ← Prediction form
│   ├── dashboard.html      ← Model performance charts
│   └── about.html          ← Project info
│
├── 1_preprocessing.py       ← EDA + outlier removal + normalization
├── 2_classification.py      ← 6 ML classifiers + comparison plots
├── 3_clustering.py          ← K-Means, DBSCAN, Hierarchical, Apriori
├── 4_deep_learning.py       ← MLP Neural Network (128→64→32)
├── 5_association_rules.py   ← Apriori ARM from scratch
├── app.py                   ← Flask web application
├── run_all.py               ← Master runner (train all + launch app)
└── requirements.txt
```

---

## 🚀 How to Run

### Step 1 – Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 – Run everything (recommended)
```bash
python run_all.py
```
This trains all models, generates all plots, and launches the web app at:
**http://127.0.0.1:5000**

### Step 3 – Or run scripts individually
```bash
python 1_preprocessing.py      # Data cleaning + EDA plots
python 2_classification.py     # Classification models
python 3_clustering.py         # Clustering algorithms
python 4_deep_learning.py      # Deep learning (MLP)
python 5_association_rules.py  # Association rule mining
python app.py                  # Launch web app
```

---

## 📊 Dataset

| Property        | Value                              |
|-----------------|------------------------------------|
| Source          | Crop_Recommendation.xlsx           |
| Rows            | 2,200                              |
| Features        | 7 (N, P, K, Temp, Humidity, pH, Rainfall) |
| Target Classes  | 22 crop types                      |
| Class Balance   | Perfectly balanced (100 each)      |
| Missing Values  | None                               |

---

## 🤖 Models & Performance

### Classification (4 core features: N, P, K, Temperature)

| Model               | Accuracy | F1-Score | CV Accuracy |
|---------------------|----------|----------|-------------|
| Random Forest       | ~86%     | ~85%     | ~84%        |
| Naive Bayes         | ~84%     | ~83%     | ~82%        |
| Decision Tree       | ~84%     | ~82%     | ~80%        |
| SVM                 | ~82%     | ~81%     | ~80%        |
| K-Nearest Neighbors | ~80%     | ~79%     | ~77%        |
| Logistic Regression | ~74%     | ~73%     | ~72%        |

> **Note:** 4-feature subset (N, P, K, Temperature) is used to simulate
> real-world partial-sensor data and avoid trivially perfect models.

### Deep Learning (MLP Neural Network)
- Architecture: Input(4) → Dense(128, ReLU) → Dense(64, ReLU) → Dense(32, ReLU) → Output(22)
- Accuracy: ~84-85%  |  ROC-AUC: ~0.99

### Unsupervised Clustering
- K-Means (k=22), DBSCAN, Hierarchical (Ward linkage), Apriori-style
- Metrics: Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz

### Association Rule Mining
- Algorithm: Apriori (implemented from scratch)
- min_support=0.04, min_confidence=0.50
- Total rules generated: 700+  |  Max Lift: ~22

---

## 🌐 Web Application

The Flask web app has 4 pages:

1. **Home** – Project overview and crop gallery
2. **Predict** – Enter soil values → get AI recommendation
3. **Dashboard** – All model metrics, confusion matrices, charts
4. **About** – Project info, pipeline, technology stack

---

## 🛠 Technologies

- Python 3.x
- scikit-learn (ML models)
- pandas / numpy (data handling)
- matplotlib / seaborn (visualization)
- Flask (web framework)
- HTML5 / CSS3 / JavaScript (frontend)
- Fraunces + DM Sans (typography)

---

