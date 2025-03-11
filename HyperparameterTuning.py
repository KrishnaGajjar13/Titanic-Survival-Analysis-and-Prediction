#!/usr/bin/env python
import os
import json
import time
import inspect
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                             roc_auc_score, roc_curve, precision_recall_curve)
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier, plot_importance

warnings.filterwarnings('ignore')

# Create results folder if it doesn't exist
os.makedirs("results", exist_ok=True)

# Dictionary to store performance metrics for all models
performance_metrics = {}

# ------------------------------------
# Load Dataset
# ------------------------------------
print("Loading dataset...")
train = pd.read_csv("Dataset/train.csv")
df = train.copy()

# ------------------------------------
# Feature Engineering
# ------------------------------------
def get_title(name):
    if "." in name:
        return name.split(",")[1].split(".")[0].strip()
    return "Unknown"

def title_map(title):
    return {"Mr": 1, "Master": 3, "Ms": 4, "Mlle": 4, "Miss": 4, "Mme": 5, "Mrs": 5}.get(title, 2)

df["title"] = df["Name"].apply(get_title).apply(title_map)
df["Fare"] = df["Fare"] > df["Fare"].mean()  # Convert Fare to Boolean (above or below average)
df.drop(["PassengerId", "Name", "Ticket"], axis=1, inplace=True)
df["Sex"] = df["Sex"].replace(["male", "female"], [0, 1])
df["Cabin"] = df["Cabin"].isna()  # True if Cabin is missing
df["Age"].fillna(df["Age"].mean(), inplace=True)
df = pd.get_dummies(df)

# ------------------------------------
# Prepare Data for Training
# ------------------------------------
print("Preparing data for training...")
y = df["Survived"]
df.drop("Survived", axis=1, inplace=True)
x = df

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)

# ------------------------------------
# Feature Scaling for Logistic Regression & XGBoost
# ------------------------------------
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# ------------------------------------
# Model Evaluation Function
# ------------------------------------
def evaluate_model(model, model_name, x_train, x_test, y_train, y_test, scaled=False):
    start_time = time.time()
    model.fit(x_train, y_train)
    train_time = time.time() - start_time

    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    with open("results/model_performance.txt", "a") as f:
        f.write(f"Model: {model_name} (Scaled: {scaled})\n")
        f.write(f"Training Time: {train_time:.2f} seconds\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"AUC: {auc:.4f}\n")
        f.write("Confusion Matrix:\n")
        f.write(f"{cm}\n")
        f.write("Classification Report:\n")
        f.write(f"{report}\n\n")
    
    performance_metrics[model_name] = {
        "scaled": scaled,
        "training_time_sec": train_time,
        "accuracy": acc,
        "auc": auc,
        "confusion_matrix": cm.tolist(),
        "classification_report": report
    }
    
    # Save ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure()
    plt.plot(fpr, tpr, marker='.', label=f"{model_name} (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend()
    plt.savefig(f"results/roc_curve_{model_name}.png")
    plt.close()

    # Save Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    plt.figure()
    plt.plot(recall, precision, marker='.', label=model_name)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve - {model_name}")
    plt.legend()
    plt.savefig(f"results/precision_recall_curve_{model_name}.png")
    plt.close()

# ------------------------------------
# Train and Evaluate Baseline Models (Without Tuning)
# ------------------------------------
print("Training RandomForestClassifier (default)...")
rf = RandomForestClassifier()
evaluate_model(rf, "RandomForest_Default", x_train, x_test, y_train, y_test)

print("Training XGBoost (default)...")
gxboost = XGBClassifier(eval_metric='logloss')
evaluate_model(gxboost, "XGBoost_Default", x_train_scaled, x_test_scaled, y_train, y_test, scaled=True)

print("Training LogisticRegression (default)...")
log_reg = LogisticRegression()
evaluate_model(log_reg, "LogisticRegression_Default", x_train_scaled, x_test_scaled, y_train, y_test, scaled=True)

# ------------------------------------
# Hyperparameter Tuning for RandomForest using RandomizedSearchCV
# ------------------------------------
print("Performing hyperparameter tuning for RandomForestClassifier...")
param_dist = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}
rf_tune = RandomForestClassifier(random_state=10)
rf_random = RandomizedSearchCV(estimator=rf_tune,
                               param_distributions=param_dist,
                               n_iter=20,
                               cv=5,
                               verbose=2,
                               random_state=10,
                               n_jobs=-1)
rf_random.fit(x_train, y_train)
print("Best parameters found for RandomForest:", rf_random.best_params_)
best_rf = rf_random.best_estimator_
evaluate_model(best_rf, "RandomForest_Tuned", x_train, x_test, y_train, y_test)

# ------------------------------------
# New Concept: Hyperparameter Tuning for XGBoost with GridSearchCV and Early Stopping
# ------------------------------------
print("Performing hyperparameter tuning for XGBoost with GridSearchCV and early stopping...")
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
param_grid_xgb = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200, 300],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

grid_search_xgb = GridSearchCV(xgb, param_grid_xgb, scoring='roc_auc', cv=3, verbose=1, n_jobs=-1)

# Check if early_stopping_rounds is accepted by the current XGBClassifier version
fit_signature = inspect.signature(xgb.fit).parameters
if 'early_stopping_rounds' in fit_signature:
    print("Early stopping supported. Running grid search with early stopping...")
    grid_search_xgb.fit(x_train_scaled, y_train, 
                        eval_set=[(x_test_scaled, y_test)], 
                        early_stopping_rounds=10, 
                        verbose=False)
else:
    print("Early stopping not supported in the current XGBClassifier version. Running grid search without early stopping...")
    grid_search_xgb.fit(x_train_scaled, y_train)

print("Best parameters for XGBoost:", grid_search_xgb.best_params_)
best_xgb = grid_search_xgb.best_estimator_
evaluate_model(best_xgb, "XGBoost_Tuned", x_train_scaled, x_test_scaled, y_train, y_test, scaled=True)

# ------------------------------------
# Cross-Validation for RandomForest_Default
# ------------------------------------
print("Performing 10-fold cross-validation for RandomForest_Default...")
cv_scores = cross_val_score(rf, x, y, cv=10, scoring='accuracy')
cv_text = f"RandomForest_Default 10-Fold CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n\n"
with open("results/model_performance.txt", "a") as f:
    f.write(cv_text)
performance_metrics["RandomForest_Default_CV"] = {
    "cv_mean_accuracy": cv_scores.mean(),
    "cv_std_accuracy": cv_scores.std()
}

# ------------------------------------
# Stacking Model: Combining Default Models
# ------------------------------------
print("Training StackingClassifier...")
stacking_model = StackingClassifier(estimators=[('rf', rf), ('xgb', gxboost)],
                                    final_estimator=LogisticRegression())
evaluate_model(stacking_model, "StackedModel", x_train_scaled, x_test_scaled, y_train, y_test, scaled=True)

# ------------------------------------
# Save performance metrics as JSON
# ------------------------------------
with open("results/model_performance.json", "w") as f:
    json.dump(performance_metrics, f, indent=4)

# ------------------------------------
# Plot Feature Importance for Tuned XGBoost
# ------------------------------------
print("Plotting feature importance for tuned XGBoost...")
best_xgb.fit(x_train_scaled, y_train)
plt.figure(figsize=(8, 6))
plot_importance(best_xgb)
plt.title("XGBoost_Tuned Feature Importance")
plt.savefig("results/xgboost_tuned_feature_importance.png")
plt.close()

print("Model evaluation complete! Results saved in 'results/' folder.")
