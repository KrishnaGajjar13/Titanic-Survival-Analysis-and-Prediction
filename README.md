# Titanic-Survival-Analysis-and-Prediction

---

```markdown
# Titanic Survival Prediction

![Titanic](https://i.pinimg.com/564x/98/73/c6/9873c68f08671ca72aece2d1ceb6b93b.jpg)

Titanic Survival Prediction is a machine learning project that uses the infamous Titanic disaster dataset to predict the survival of passengers. This repository is designed for beginners to get started with Kaggle competitions and supervised learning. It covers data loading, cleaning, feature engineering, model training (including hyperparameter tuning), evaluation, and results interpretation.

---

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Modeling & Tuning](#modeling--tuning)
- [Results & Analysis](#results--analysis)
- [Tools & Libraries](#tools--libraries)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Overview

This project aims to predict whether a passenger survived the Titanic disaster using various machine learning models. The repository demonstrates:
- Data exploration and cleaning
- Feature engineering (including extracting titles from names)
- Model building using Logistic Regression, Random Forest, and XGBoost
- Hyperparameter tuning using RandomizedSearchCV and GridSearchCV (with early stopping for XGBoost)
- Ensemble methods through model stacking
- Evaluation metrics including accuracy, ROC-AUC, and Precision-Recall curves

The project is intended for anyone beginning their machine learning journey and covers each step in a detailed yet beginner-friendly manner.

---

## Repository Structure

```
Titanic-Survival-Prediction/
├── Dataset/
│   ├── train.csv         # Training dataset
│   ├── test.csv          # Testing dataset
├── results/
│   ├── model_performance.txt  # Evaluation logs for different models
│   ├── model_performance.json # JSON summary of model performance metrics
│   ├── roc_curve_*.png        # ROC curve images for each model
│   ├── precision_recall_curve_*.png  # Precision-Recall curve images for each model
│   ├── xgboost_tuned_feature_importance.png  # Feature importance for tuned XGBoost
├── Analysis.py  # Standalone script with hyperparameter tuning examples
├── main.py
├── HyperparameterTuning.py
├── README.md                # Project overview and instructions (this file)
└── LICENSE.txt              # License information
```

---

## Installation

### Requirements
- Python 3.12

### Libraries
Install required libraries using pip:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost 
```

### Running the Project
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/Titanic-Survival-Prediction.git
   cd Titanic-Survival-Prediction
   ```
2. **Run the Standalone Script:**
   Alternatively, you can run the script with hyperparameter tuning:
   ```bash
   python main.py
   ```

---

## Usage

The project demonstrates the entire modeling workflow:
- **Data Loading & Cleaning:** Handling missing values and removing irrelevant columns.
- **Feature Engineering:** Extracting titles from names, encoding categorical features, and more.
- **Model Training:** Training baseline models and then applying hyperparameter tuning.
- **Model Evaluation:** Saving metrics (accuracy, AUC, confusion matrix, etc.), and plotting ROC and Precision-Recall curves.
- **Ensembling:** Combining predictions using a stacking classifier.

The results are saved in the `results/` folder for review and further analysis.

---

## Modeling & Tuning

### Baseline Models
The project builds baseline models using:
- **RandomForestClassifier**
- **XGBoost**
- **LogisticRegression**

### Hyperparameter Tuning
Two tuning approaches are demonstrated:
- **RandomizedSearchCV for RandomForest:** Tuning parameters like number of estimators, max depth, and more.
- **GridSearchCV for XGBoost with Early Stopping:**  
  This advanced tuning method leverages GridSearchCV along with XGBoost's early stopping feature (if supported by your version).  
  _Note:_ If your installed version of XGBoost does not support `early_stopping_rounds`, the grid search will run without it.

---

## Results & Analysis

The project saves comprehensive performance logs and plots:
- **model_performance.txt:** Contains accuracy, AUC, training time, and classification reports.
- **model_performance.json:** JSON summary of performance metrics.
- **ROC and Precision-Recall Curves:** Visual evaluation of model performance.
- **Feature Importance:** Visualization for the tuned XGBoost model.

Key insights include the impact of features such as gender, age, and class on survival rates, and the effectiveness of various hyperparameter tuning strategies.

---

## Tools & Libraries

- **Python:** Primary programming language.
- **Jupyter Notebook:** For interactive development.
- **pandas & NumPy:** For data manipulation.
- **Matplotlib & Seaborn:** For data visualization.
- **scikit-learn:** For machine learning and evaluation.
- **XGBoost:** For gradient boosting and advanced tuning techniques.

---

## Contributing

Contributions are welcome! If you have suggestions, improvements, or additional features, please:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your_feature`).
3. Commit your changes (`git commit -m 'Add some feature'`).
4. Push your branch (`git push origin feature/your_feature`).
5. Open a pull request.

---

## License

This project is licensed under the terms of the [Apache License](LICENSE.txt).

---

## Contact

For questions, suggestions, or collaboration opportunities, please contact:

**Ruby Poddar**  
Email: [krishnagajjar1311@gmail.com](mailto:krishnagajjar1311@gmail.com)

---

_If you find this project useful, please consider giving it a star!_
