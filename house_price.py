# ============================================================
# House Price Prediction
# Dataset: Kaggle - House Prices: Advanced Regression Techniques
# Models: Linear Regression, Random Forest, XGBoost
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# ============================================================
# 1. LOAD DATA
# ============================================================
train = pd.read_csv("data/train.csv")
print("Train Dataset loaded! Shape: ", train.shape)

test = pd.read_csv('data/test.csv')
print("Test Dataset loaded! Shape: ", test.shape)

# ============================================================
# 2. Target Transformation
# ============================================================
train['SalePrice'] = np.log1p(train['SalePrice'])

# ============================================================
# 3. DROP HIGH MISSING COLUMNS
# ============================================================
total_rows = train.shape[0]
threshold = total_rows * 0.5
train = train.dropna(thresh=threshold, axis=1)

# Top features correlated with SalePrice
# numeric_train = train.select_dtypes(include=[np.number])
# correlation = numeric_train.corr()
# top_features_corr_SalePrice = correlation['SalePrice'].sort_values(ascending=False).head(11)
# print("\nTop correlated features:\n", top_features_corr_SalePrice)


# ============================================================
# 4. FEATURE / TARGET SPLIT
# ============================================================
x = train.drop('SalePrice', axis=1)
y = train['SalePrice']

# ============================================================
# 5. HANDLE MISSING VALUES
# ============================================================
# Fill numerical missing values with median for train dataset
numerical_col = x.select_dtypes(include=[np.number]).columns
train[numerical_col] = train[numerical_col].fillna(train[numerical_col].median())

# Fill categorical missing values with mode for train dataset
categorical_col = x.select_dtypes(include=['str']).columns
train[categorical_col] = train[categorical_col].fillna(train[categorical_col].mode().iloc[0])

# Fill numerical missing values with median for test dataset
test[numerical_col] = test[numerical_col].fillna(train[numerical_col].median())

# Fill categorical missing values with mode for test dataset
test[categorical_col] = test[categorical_col].fillna(train[categorical_col].mode().iloc[0])

# print("\nData cleaned! Shape: ", train.shape)
# print("\nMissing values remaining: ", train.isnull().sum().sum())

# ============================================================
# 6. ENCODING
# ============================================================
train = pd.get_dummies(train)
test = pd.get_dummies(test)

# Align columns
x = train.drop('SalePrice', axis=1)
print("\nEncoding done! Shape: ", train.shape)

# ============================================================
# 7 . MODELING
# ============================================================

results = {}

# Linear Regression
lr_model = LinearRegression()
scores_lr = cross_val_score(lr_model, x, y, cv=5, scoring='r2')
y_pred_lr = cross_val_predict(lr_model, x, y, cv=5)
print(f"Linear Regression CV R2: {scores_lr.mean():.3f}")
results['Linear Regression'] = {
    "MAE": mean_absolute_error(np.expm1(y), np.expm1(y_pred_lr)),
    "RMSE": np.sqrt(mean_squared_error(np.expm1(y), np.expm1(y_pred_lr))),
    "R2 Score": r2_score(y, y_pred_lr)
}

# Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

rf_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5]
}

rf_grid = GridSearchCV(
    RandomForestRegressor(random_state=42),
    rf_param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1
)

rf_grid.fit(x, y)
rf_best_model = rf_grid.best_estimator_
print(f"RF Best Params: {rf_grid.best_params_}", )
print(f"RF Best Score: {rf_grid.best_score_}")

y_pred_rf = cross_val_predict(rf_best_model,x,y, cv=5)
results['Random Forest'] = {
    "MAE": mean_absolute_error(np.expm1(y), np.expm1(y_pred_rf)),
    "RMSE": np.sqrt(mean_squared_error(np.expm1(y), np.expm1(y_pred_rf))),
    "R2 Score": r2_score(y, y_pred_rf)
}

# XGBoost
xgb_model = XGBRegressor(n_estimators=100, random_state=42)
scores_xgb = cross_val_score(xgb_model, x, y, cv=5, scoring='r2')
print("XGBoost CV R2: ", scores_xgb.mean())
xgb_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1]
}
xgb_grid = GridSearchCV(
    XGBRegressor(random_state=42),
    xgb_param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1
)

xgb_grid.fit(x, y)

xgb_best_model = xgb_grid.best_estimator_
print(f"XGBoost Best Params: {xgb_grid.best_params_}")
print(f"XGBoost Best Score: {xgb_grid.best_score_}")

y_pred_xgb = cross_val_predict(xgb_best_model,x,y, cv=5)
results['XGBoost'] = {
    "MAE": mean_absolute_error(np.expm1(y), np.expm1(y_pred_xgb)),
    "RMSE": np.sqrt(mean_squared_error(np.expm1(y), np.expm1(y_pred_xgb))),
    "R2 Score": r2_score(y, y_pred_xgb)
}

 # Print results
print("\n===== MODEL COMPARISON =====")
for model_name, metrics in results.items():
    print(f"\n{model_name}:")
    print(f"  MAE: ${metrics['MAE']:,.2f}")
    print(f"  RMSE: ${metrics['RMSE']:,.2f}")
    print(f"  R2: {metrics['R2 Score']:.4f}")

# ============================================================
# 9. VISUALIZATION
# ============================================================
# Histogram plot for SalePrice column (Checks number of houses within available price range)
plt.figure(figsize=(8,5))
sns.histplot(train['SalePrice'], kde=True, color='black')
plt.title('Distribution of Sale Price after Log Transformation')
plt.show()

# Heatmap to show correlation between columns
plt.figure(figsize=(12,8))
sns.heatmap(train[numerical_col].corr(), cmap='coolwarm')
plt.title("Correlation Heat Map")
plt.show()

# Actual vs. Predicted Prices
plt.figure(figsize=(6,6))
plt.scatter(y, y_pred_xgb, alpha=0.5)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices(XGBoost)")

# Perfect prediction line
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red')
plt.show()

# Important features plot
importance = xgb_best_model.feature_importances_

features = x.columns
feat_imp = pd.DataFrame({
    'feature': features,
    'importance': importance
})

feat_imp = feat_imp.sort_values(by='importance', ascending=False)

top_features = feat_imp.head(10)

plt.figure()
plt.barh(top_features['feature'], top_features['importance'])
plt.gca().invert_yaxis()
plt.title("Top 10 Important Features")
plt.savefig("plots/top_features.png", bbox_inches="tight")
plt.show()

# Scatterplot for showing relationship between SalePrice and top columns which had high correlation with SalePrice
fig, axes = plt.subplots(2, 3, figsize=(15,10))
axes = axes.flatten()
top_6_features = feat_imp.head(6)
for i, feature in enumerate(top_6_features['feature']):
    axes[i].scatter(train[feature], train['SalePrice'], alpha=0.5)
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel('SalePrice')
    axes[i].set_title(f'{feature} vs SalePrice')
plt.tight_layout()
plt.show()


# ============================================================
# 10. KAGGLE SUBMISSION
# ============================================================

test_ids = test['Id']

# Drop columns with more than 50% missing values
test = test.dropna(thresh=threshold, axis=1)

# One hot encoding

# Align test columns with train columns (very important!)
test = test.reindex(columns=x.columns, fill_value=0)

# Predict using XGBoost
test_predictions = np.expm1(xgb_best_model.predict(test))

# Submission file
submission = pd.DataFrame({
    'Id': test_ids,
    'SalePrice': test_predictions
})
submission.to_csv('submission.csv', index=False)
print(submission.head())