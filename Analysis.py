# ========================
# Importing Required Libraries
# ========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# Sklearn Modules
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV

# XGBoost Modules
from xgboost import XGBClassifier, plot_importance

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# ========================
# Load Dataset
# ========================
train = pd.read_csv("Dataset/train.csv")
test = pd.read_csv("Dataset/test.csv")
df = train

# ========================
# Data Overview
# ========================
df.info()
print("\nStatistics of Numeric Columns:")
print(df.describe())

# ========================
# Survival Analysis
# ========================
print("\nSurvived Members Analysis:")
ab = df.groupby("Survived").mean(numeric_only=True)
print(ab)

# Compute Absolute Differences
v1 = ab.iloc[0, :]
v2 = ab.iloc[1, :]
abs_diff = abs((v2 - v1) / (v1 + v2))

# ========================
# Visualizations
# ========================

# Histogram of Age by Survival Status
g = sns.FacetGrid(df, col="Survived")
g.map(plt.hist, "Age", bins=20)
plt.savefig("static/age_survival_hist.png")  # Save image
plt.show()

# Histogram of Age by Survival Status & Pclass
grid = sns.FacetGrid(df, col="Survived", row="Pclass", height=2.2, aspect=1.6)
grid.map(plt.hist, "Age", alpha=0.5, bins=20)
grid.add_legend()
plt.savefig("static/age_pclass_survival.png")  # Save image
plt.show()

# Survival Rate by Pclass, Sex, and Embarked
grid = sns.FacetGrid(df, row="Embarked", height=2.2, aspect=1.6)
grid.map(sns.pointplot, "Pclass", "Survived", "Sex", palette="deep")
grid.add_legend()
plt.savefig("static/survival_rate_pclass_sex_embarked.png")  # Save image
plt.show()

# Fare Distribution by Sex and Survival Status
grid = sns.FacetGrid(df, row="Embarked", col="Survived", height=2.2, aspect=1.6)
grid.map(sns.barplot, "Sex", "Fare", alpha=0.5, ci=None)
grid.add_legend()
plt.savefig("static/fare_distribution.png")  # Save image
plt.show()

# Correlation Heatmap
corr = df.corr(numeric_only=True)
plt.figure(figsize=(8, 8))
sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, cmap="coolwarm")
plt.savefig("static/correlation_heatmap.png")  # Save image
plt.show()