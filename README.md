# Quality-Classifier
# 🍷 Wine Quality Classification

This project is a **binary classification model** that predicts whether a wine sample is of *"best quality"* based on its physicochemical features. The goal is to demonstrate end-to-end data science workflow using a real dataset.

---

## 📊 Problem Statement

Given a dataset of red and white wines with features such as acidity, sugar, pH, alcohol, etc., classify each wine into:
- `1`: Best quality (quality > 5)
- `0`: Not best quality (quality ≤ 5)

---

## 🧠 Models Used

Three popular classification models are trained and compared:
- **Logistic Regression**
- **XGBoost Classifier**
- **Support Vector Machine (SVC with RBF Kernel)**

Performance is measured using:
- ROC-AUC Score
- Confusion Matrix (for XGBoost)
- Classification Report (for XGBoost)

---

## 📁 Dataset

Dataset: [UCI Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/Wine+Quality)

You can combine both red and white wine datasets into a single CSV named:
data/winequality.csv

yaml
Copy
Edit

---

## 🛠️ Project Structure

wine-quality-classifier/
├── data/
│ └── winequality.csv # Dataset (not included in repo)
├── wine_quality_classifier.py # Fully commented and runnable Python script
├── requirements.txt # Python package dependencies
└── README.md # This file

yaml
Copy
Edit

---

## ▶️ How to Run

1. **Clone this repo**:
```bash
git clone https://github.com/yourusername/wine-quality-classifier.git
cd wine-quality-classifier
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Run the script:

bash
Copy
Edit
python wine_quality_classifier.py
Make sure the dataset file winequality.csv is placed in the data/ folder.

✅ Output
You’ll get:

Training & validation ROC-AUC scores for all models

Confusion matrix and classification report for XGBoost

Histograms, correlation heatmaps, and alcohol vs. quality bar chart

📌 Key Notes
Target variable is engineered as best quality = 1 if quality > 5.

Data is normalized using MinMaxScaler.

Missing values are imputed using column mean.

total sulfur dioxide is dropped due to high correlation.
