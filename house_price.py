# ============================================================
# House Price Prediction
# Dataset: Kaggle - House Prices: Advanced Regression Techniques
# Models: Linear Regression, Random Forest, XGBoost
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# ============================================================
# 1. LOAD DATA
# ============================================================
train = pd.read_csv("data/train.csv")
print("Dataset loaded! Shape: ", train.shape)

# ============================================================
# 2. EDA (Exploratory Data Analysis)
# ============================================================

# Log transform SalePrice to fix right skew
train['SalePrice'] = np.log1p(train['SalePrice'])

# Top features correlated with SalePrice
numeric_train = train.select_dtypes(include=[np.number])
correlation = numeric_train.corr()
top_features = correlation['SalePrice'].sort_values(ascending=False).head(11)
print("\nTop correlated features:\n", top_features)

# ============================================================
# 3. DATA CLEANING
# ============================================================

# Drop columns with more than 50% missing values
total_rows = train.shape[0]
threshold = total_rows * 0.5
train = train.dropna(thresh=threshold, axis=1)

# Fill numerical missing values with median
numerical_col = train.select_dtypes(include=[np.number]).columns
train[numerical_col] = train[numerical_col].fillna(train[numerical_col].median())

# Fill categorical missing values with mode
categorical_col = train.select_dtypes(include=['str']).columns
train[categorical_col] = train[categorical_col].fillna(train[categorical_col].mode().iloc[0])

print("\nData cleaned! Shape: ", train.shape)
print("\nMissing values remaining: ", train.isnull().sum().sum())

# ============================================================
# 4. ENCODING
# ============================================================

# One hot encoding categorical columns
train = pd.get_dummies(train)
print("\nEncoding done! Shape: ", train.shape)

# ============================================================
# 5. FEATURE SELECTION
# ============================================================

# Separate features and target
x = train.drop('SalePrice', axis=1)
y = train['SalePrice']

# Split into train and test sets (80/20)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print("\nTraining data: ", x_train.shape)
print("Testing data:", x_test.shape)

# ============================================================
# 6. MODELING
# ============================================================

results = {}

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(x_train, y_train)
y_pred_lr = lr_model.predict(x_test)
results['Linear Regression'] = {
    "MAE": mean_absolute_error(np.expm1(y_test), np.expm1(y_pred_lr)),
    "RMSE": np.sqrt(mean_squared_error(np.expm1(y_test), np.expm1(y_pred_lr))),
    "R2 Score": r2_score(np.expm1(y_test), np.expm1(y_pred_lr))
}

# Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(x_train, y_train)
y_pred_rf = rf_model.predict(x_test)
results['Random Forest'] = {
    "MAE": mean_absolute_error(np.expm1(y_test), np.expm1(y_pred_rf)),
    "RMSE": np.sqrt(mean_squared_error(np.expm1(y_test), np.expm1(y_pred_rf))),
    "R2 Score": r2_score(np.expm1(y_test), np.expm1(y_pred_rf))
}

# XGBoost
xgb_model = XGBRegressor(n_estimators=100, random_state=42)
xgb_model.fit(x_train, y_train)
y_pred_xgb = xgb_model.predict(x_test)
results['XGBoost'] = {
    "MAE": mean_absolute_error(np.expm1(y_test), np.expm1(y_pred_xgb)),
    "RMSE": np.sqrt(mean_squared_error(np.expm1(y_test), np.expm1(y_pred_xgb))),
    "R2 Score": r2_score(np.expm1(y_test), np.expm1(y_pred_xgb))
}

# Print results
print("\n===== MODEL COMPARISON =====")
for model_name, metrics in results.items():
    print(f"\n{model_name}:")
    print(f"  MAE: ${metrics['MAE']:,.2f}")
    print(f"  RMSE: ${metrics['RMSE']:,.2f}")
    print(f"  R2: {metrics['R2 Score']:.4f}")

# ============================================================
# 7. VISUALIZATION
# ============================================================

# Actual vs Predicted (XGBoost)
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred_xgb, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices (XGBoost)")
plt.tight_layout()
plt.savefig('plots/actual_vs_predicted.png', bbox_inches='tight')

# Feature Importance
importance = pd.Series(xgb_model.feature_importances_, index=x_train.columns)
importance = importance.sort_values(ascending=False).head(15)
plt.figure(figsize=(10,6))
importance.plot(kind='bar')
plt.title('Top 15 Most Important Features')
plt.ylabel('Importance Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('plots/feature_importance.png', bbox_inches='tight')

# ============================================================
# 8. KAGGLE SUBMISSION
# ============================================================

# Load test data
test = pd.read_csv('data/test.csv')
test_ids = test['Id']

# Drop columns with more than 50% missing values
test = test.dropna(thresh=threshold, axis=1)

# Fill numerical missing values with median
numerical_col_test = test.select_dtypes(include=[np.number]).columns
test[numerical_col_test] = test[numerical_col_test].fillna(test[numerical_col_test].median())

# Fill categorical missing values with mode
categorical_col_test = test.select_dtypes(include=['str']).columns
test[categorical_col_test] = test[categorical_col_test].fillna(test[categorical_col_test].mode().iloc[0])

# One hot encoding
test = pd.get_dummies(test)

# Align test columns with train columns (very important!)
test = test.reindex(columns=x_train.columns, fill_value=0)

# Predict using linear regression
test_predictions = np.expm1(lr_model.predict(test))

# Submission file
submission = pd.DataFrame({
    'Id': test_ids,
    'SalePrice': test_predictions
})
submission.to_csv('submission.csv', index=False)
print(submission.head())