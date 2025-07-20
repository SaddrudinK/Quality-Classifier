"""
Wine Quality Classification Project

This script performs binary classification on a wine quality dataset.
Steps included:
1. Data loading and inspection
2. Data cleaning and preprocessing
3. Feature transformation and target creation
4. Model training (Logistic Regression, XGBoost, SVM)
5. Evaluation with ROC-AUC, confusion matrix, and classification report

"""

# 1. Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn import metrics
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import warnings
warnings.filterwarnings('ignore')  # Suppress warnings

# 2. Load dataset
df = pd.read_csv('data/winequality.csv')  # Adjust path if needed
print("First 5 rows:")
print(df.head())
print("\nInfo:")
print(df.info())
print("\nStatistics:")
print(df.describe().T)

# 3. Check and fill missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Fill missing values with column mean
for col in df.columns:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mean())

print("Remaining missing values:", df.isnull().sum().sum())

# 4. Visualizations
df.hist(bins=20, figsize=(10, 10))
plt.tight_layout()
plt.show()

plt.bar(df['quality'], df['alcohol'])
plt.xlabel('Quality')
plt.ylabel('Alcohol')
plt.title('Alcohol Content by Quality')
plt.show()

# 5. Convert object types to numeric if any
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = pd.to_numeric(df[col], errors='coerce')

# 6. Correlation analysis
plt.figure(figsize=(12, 12))
sb.heatmap(df.corr() > 0.7, annot=True, cbar=False)
plt.title("Feature Correlation > 0.7")
plt.show()

# 7. Feature selection and target creation
if 'total sulfur dioxide' in df.columns:
    df = df.drop('total sulfur dioxide', axis=1)

# Binary target: quality > 5 = 1, else 0
df['best quality'] = [1 if x > 5 else 0 for x in df['quality']]

# Convert wine type (if present)
df.replace({'white': 1, 'red': 0}, inplace=True)

# 8. Feature/target split
features = df.drop(['quality', 'best quality'], axis=1)
target = df['best quality']

# 9. Train-test split
xtrain, xtest, ytrain, ytest = train_test_split(
    features, target, test_size=0.2, random_state=40
)

# 10. Imputation
imputer = SimpleImputer(strategy='mean')
xtrain = imputer.fit_transform(xtrain)
xtest = imputer.transform(xtest)

# 11. Normalization
scaler = MinMaxScaler()
xtrain = scaler.fit_transform(xtrain)
xtest = scaler.transform(xtest)

# 12. Model training and evaluation
models = [
    LogisticRegression(),
    XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    SVC(kernel='rbf', probability=True)
]

for model in models:
    model.fit(xtrain, ytrain)
    model_name = model.__class__.__name__
    print(f"\n{model_name}:")

    # Evaluate using ROC-AUC
    if hasattr(model, 'predict_proba'):
        train_auc = metrics.roc_auc_score(ytrain, model.predict_proba(xtrain)[:, 1])
        val_auc = metrics.roc_auc_score(ytest, model.predict_proba(xtest)[:, 1])
    else:
        train_auc = metrics.roc_auc_score(ytrain, model.predict(xtrain))
        val_auc = metrics.roc_auc_score(ytest, model.predict(xtest))

    print(f"Training ROC-AUC Score   : {train_auc:.4f}")
    print(f"Validation ROC-AUC Score : {val_auc:.4f}")

# 13. Confusion Matrix for XGBoost
xgb_model = models[1]
cm = confusion_matrix(ytest, xgb_model.predict(xtest))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
disp.plot()
plt.title("Confusion Matrix - XGBoost")
plt.show()

# 14. Classification report
print("\nClassification Report - XGBoost:")
print(metrics.classification_report(ytest, xgb_model.predict(xtest)))
