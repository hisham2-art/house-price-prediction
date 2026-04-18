# 🏠 House Price Prediction

A machine learning project to predict house prices using advanced regression techniques. This project uses multiple models including Linear Regression, Random Forest, and XGBoost, with a focus on proper data preprocessing, cross-validation, and model evaluation.

---

## 📌 Project Overview

The goal of this project is to accurately predict house prices based on various features such as size, quality, number of rooms, and more.

This project demonstrates:

* Data cleaning and preprocessing
* Feature engineering and encoding
* Model training and evaluation
* Hyperparameter tuning
* Cross-validation for reliable performance
* Kaggle submission pipeline

---

## 📊 Dataset

* Source: Kaggle – *House Prices: Advanced Regression Techniques*
* Training Data: 1460 samples
* Test Data: 1459 samples
* Features: 80+ variables describing residential homes

---

## ⚙️ Workflow

### 1. Data Preprocessing

* Removed columns with more than 50% missing values
* Filled missing values:

  * Numerical → Median
  * Categorical → Mode
* Applied **log transformation** on target variable (`SalePrice`) to reduce skewness

---

### 2. Feature Engineering

* One-hot encoding for categorical variables
* Feature alignment between train and test datasets

---

### 3. Model Training

The following models were used:

* Linear Regression
* Random Forest Regressor
* XGBoost Regressor

---

### 4. Model Evaluation

Models were evaluated using:

* **R² Score** → Model fit
* **MAE (Mean Absolute Error)** → Average error
* **RMSE (Root Mean Squared Error)** → Penalizes large errors

Cross-validation was used to ensure reliable and unbiased performance.

---

## 📈 Results

| Model             | MAE ($) | RMSE ($) | R² Score |
| ----------------- | ------- | -------- | -------- |
| Linear Regression | 17,061  | 63,440   | 0.36     |
| Random Forest     | 17,601  | 30,434   | 0.85     |
| XGBoost           | 15,901  | 27,868   | 0.87     |

✅ **Best Model: XGBoost**

---

## 📊 Feature Importance

Top features influencing house prices:

![Top Features](plots/top_features.png)

---

## 🚀 Kaggle Performance

* Score improved from **0.155 → 0.134**
* Achieved using:

  * Cross-validation
  * Hyperparameter tuning
  * Log transformation

---

## 🧠 Key Learnings

* Cross-validation provides realistic model evaluation
* Log transformation helps with skewed target variables
* Tree-based models handle non-linearity better than linear models
* Consistent preprocessing between train and test is critical
* MAE and RMSE provide better real-world insight than R² alone

---

## 📂 Project Structure

```
house-price-prediction/
│
├── data/
│   ├── train.csv
│   └── test.csv
│
├── plots/
│   └── top_features.png
│
├── house_price.py
├── submission.csv
├── requirements.txt
└── README.md
```

---

## 🛠️ Installation

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

```bash
python house_price.py
```

This will:

* Train models
* Evaluate performance
* Generate predictions
* Create `submission.csv`

---

## 📦 Requirements

* pandas
* numpy
* matplotlib
* scikit-learn
* xgboost

---

## 🎯 Future Improvements

* Feature engineering (e.g., total area, house age)
* Advanced encoding techniques (target encoding)
* Model ensembling
* Pipeline automation
* Deployment using Streamlit

---

## 🙌 Acknowledgements

* Kaggle for dataset and competition
* Scikit-learn & XGBoost for ML tools

---

## 📬 Contact

If you found this project useful or have suggestions, feel free to reach out!

---
