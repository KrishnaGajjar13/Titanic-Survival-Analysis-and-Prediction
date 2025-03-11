import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import json
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, 
    roc_auc_score, roc_curve, precision_recall_curve
)
import os, xgboost
from xgboost import XGBClassifier, plot_importance
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# Load Dataset
train = pd.read_csv("Dataset/train.csv")
test = pd.read_csv("Dataset/test.csv")
df = train

# Feature Engineering
def get_title(name):
    if "." in name:
        return name.split(",")[1].split(".")[0].strip()
    return "Unknown"

def title_map(title):
    return {"Mr": 1, "Master": 3, "Ms": 4, "Mlle": 4, "Miss": 4, "Mme": 5, "Mrs": 5}.get(title, 2)

df["title"] = df["Name"].apply(get_title).apply(title_map)
df["Fare"] = df["Fare"] > df["Fare"].mean()
df.drop(["PassengerId", "Name", "Ticket"], axis=1, inplace=True)
df["Sex"] = df["Sex"].replace(["male", "female"], [0, 1])
df["Cabin"] = df["Cabin"].isna()
df["Age"].fillna(df["Age"].mean(), inplace=True)
df = pd.get_dummies(df)

# Prepare Data for Training
y = df["Survived"]
df.drop("Survived", axis=1, inplace=True)
x = df
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)

# Feature Scaling for Logistic Regression & XGBoost
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Model Evaluation Function
def evaluate_model(model, model_name, x_train, x_test, y_train, y_test, scaled=False):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    with open("results/model_performance.txt", "a") as f:
        f.write(f"Model: {model_name} (Scaled: {scaled})\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"AUC: {auc:.4f}\n")
        f.write(f"Confusion Matrix:\n{cm}\n")
        f.write(f"Classification Report:\n{report}\n\n")
    
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure()
    plt.plot(fpr, tpr, marker='.', label=model_name)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(f"results/roc_curve_{model_name}.png")
    
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    plt.figure()
    plt.plot(recall, precision, marker='.', label=model_name)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.savefig(f"results/precision_recall_curve_{model_name}.png")

# Train and Evaluate Models
rf = RandomForestClassifier()
evaluate_model(rf, "RandomForest", x_train, x_test, y_train, y_test)

gxboost = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
evaluate_model(gxboost, "XGBoost", x_train_scaled, x_test_scaled, y_train, y_test, scaled=True)


log_reg = LogisticRegression()
evaluate_model(log_reg, "LogisticRegression", x_train_scaled, x_test_scaled, y_train, y_test, scaled=True)

# Cross-Validation
cv_scores = cross_val_score(rf, x, y, cv=10, scoring='accuracy')
with open("results/model_performance.txt", "a") as f:
    f.write(f"Random Forest 10-Fold Cross-Validation Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n\n")

# Stacking Model
stacking_model = StackingClassifier(estimators=[('rf', rf), ('xgb', gxboost)], final_estimator=LogisticRegression())
evaluate_model(stacking_model, "StackedModel", x_train_scaled, x_test_scaled, y_train, y_test, scaled=True)

print("Model evaluation complete! Results saved in 'results/' folder.")