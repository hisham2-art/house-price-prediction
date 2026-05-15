# ============================================================
# House Price Prediction
# Dataset: Kaggle - House Prices: Advanced Regression Techniques
# Models: Linear Regression, Random Forest, XGBoost
# ============================================================
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_val_predict, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
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
dropped_cols = train.columns[train.isnull().sum() > threshold].tolist()
train = train.drop(columns=dropped_cols)
test = test.drop(columns=dropped_cols, errors='ignore')
# Top features correlated with SalePrice
# numeric_train = train.select_dtypes(include=[np.number])
# correlation = numeric_train.corr()
# top_features_corr_SalePrice = correlation['SalePrice'].sort_values(ascending=False).head(11)
# print("\nTop correlated features:\n", top_features_corr_SalePrice)


# ============================================================
# 4. FEATURE / TARGET SPLIT
# ============================================================

x = train.drop(['SalePrice','Id'], axis=1)
y = train['SalePrice']

def engineer_features(df):
    df =df.copy()

    df['TotalSF'] = (
        df['TotalBsmtSF'].fillna(0) +
        df['1stFlrSF'].fillna(0) +
        df['2ndFlrSF'].fillna(0)
    )

    df['TotalBathrooms'] = (
        df['FullBath'].fillna(0) +
        df['HalfBath'].fillna(0) * 0.5 +
        df['BsmtFullBath'].fillna(0) +
        df['BsmtHalfBath'].fillna(0) * 0.5
    )

    df['TotalPorchSF'] = (
        df['OpenPorchSF'].fillna(0) +
        df['EnclosedPorch'].fillna(0) +
        df['3SsnPorch'].fillna(0) +
        df['ScreenPorch'].fillna(0)
    )

    df['HouseAge'] = df['YrSold'] - df['YearBuilt']

    df['YearsSinceRemodel'] = df['YrSold'] - df['YearRemodAdd']

    df['IsRemodeled'] = (df['YearRemodAdd'] != df['YearBuilt']).astype(int)
    df['HasBasement'] = (df['TotalBsmtSF'].fillna(0) > 0).astype(int)
    df['Has1stFloor'] = (df['1stFlrSF'].fillna(0) > 0).astype(int)
    df['Has2ndFloor'] = (df['2ndFlrSF'].fillna(0) > 0).astype(int)
    df['HasGarage'] = (df['GarageArea'].fillna(0) > 0).astype(int)

    df['IsNew'] = (df['HouseAge'] <= 2).astype(int)

    return df

x = engineer_features(x)
test = engineer_features(test)

# ============================================================
# 5. HANDLE MISSING VALUES
# ============================================================

# # Fill numerical missing values with median for test dataset
# test[numerical_col] = test[numerical_col].fillna(train[numerical_col].median())
#
# # Fill categorical missing values with mode for test dataset
# test[categorical_col] = test[categorical_col].fillna(train[categorical_col].mode().iloc[0])
# Select numerical and categorical missing values with median for train dataset
numerical_col = x.select_dtypes(include=[np.number]).columns.tolist()
categorical_col = x.select_dtypes(include=['str']).columns.tolist()

print(f"\nAfter feature engineering:")
print(f"Numerical features: {numerical_col}")
print(f"Categorical features: {categorical_col}")

# ============================================================
# 6. TRAIN / HOLDOUT SPLIT (Phase 2)
# ============================================================

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

print(f"Training rows: {x_train.shape[0]}")
print(f"Testing rows: {x_test.shape[0]}")


# ============================================================
# 7. PIPELINE BUILDER FUNCTION (Phase 4)
# Wrapped in a function so we can swap models cleanly
# during comparison without rewriting the preprocessor.
# Everything else stays identical — the only variable is
# the model at the end of the pipeline.
# ============================================================


def build_pipeline(model):
    preprocessor = ColumnTransformer(transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scalar', StandardScaler())
        ]), numerical_col),
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_col)
    ], remainder='drop')

    return Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

# print("\nData cleaned! Shape: ", train.shape)
# print("\nMissing values remaining: ", train.isnull().sum().sum())

# # ============================================================
# # 6. ENCODING
# # ============================================================
# train = pd.get_dummies(train)
# test = pd.get_dummies(test)

# # Align columns
# x = train.drop('SalePrice', axis=1)
# print("\nEncoding done! Shape: ", train.shape)

# ============================================================
# 8 . MODELING COMPARISON
# ============================================================

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=100, random_state=42,verbosity=0)
}

best_score = -np.inf
best_name = None

for name, model_name in models.items():
    pipeline = build_pipeline(model_name)
    scores = cross_val_score(
        pipeline, x_train, y_train, cv=5, scoring='r2'
    )
    print(f"{name:20s} → Mean R²: {scores.mean():.4f} Std: {scores.std():.4f}")
    if scores.mean() > best_score:
        best_score = scores.mean()
        best_name = name

print(f"\nBest model: {best_name} (R² = {best_score:.4f})")



# results = {}
#
# # Linear Regression
# lr_model = LinearRegression()
# scores_lr = cross_val_score(lr_model, x, y, cv=5, scoring='r2')
# y_pred_lr = cross_val_predict(lr_model, x, y, cv=5)
# print(f"Linear Regression CV R2: {scores_lr.mean():.3f}")
# results['Linear Regression'] = {
#     "MAE": mean_absolute_error(np.expm1(y), np.expm1(y_pred_lr)),
#     "RMSE": np.sqrt(mean_squared_error(np.expm1(y), np.expm1(y_pred_lr))),
#     "R2 Score": r2_score(y, y_pred_lr)
# }
#
# # Random Forest
# rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
#
# rf_param_grid = {
#     'n_estimators': [100, 200],
#     'max_depth': [5, 10, None],
#     'min_samples_split': [2, 5]
# }
#
# rf_grid = GridSearchCV(
#     RandomForestRegressor(random_state=42),
#     rf_param_grid,
#     cv=5,
#     scoring='r2',
#     n_jobs=-1
# )
#
# rf_grid.fit(x, y)
# rf_best_model = rf_grid.best_estimator_
# print(f"RF Best Params: {rf_grid.best_params_}", )
# print(f"RF Best Score: {rf_grid.best_score_}")
#
# y_pred_rf = cross_val_predict(rf_best_model,x,y, cv=5)
# results['Random Forest'] = {
#     "MAE": mean_absolute_error(np.expm1(y), np.expm1(y_pred_rf)),
#     "RMSE": np.sqrt(mean_squared_error(np.expm1(y), np.expm1(y_pred_rf))),
#     "R2 Score": r2_score(y, y_pred_rf)
# }

# ============================================================
# 9. HYPERPARAMETER TUNING (your original GridSearchCV)
# We tune only XGBoost since it consistently wins on
# tabular data. GridSearchCV tries every combination of
# parameters and picks the best one using 5-fold CV.
# n_jobs=-1 uses all CPU cores — speeds up the search.
# ============================================================
xgb_param_grid = {
    'model__n_estimators': [100, 200],
    'model__max_depth': [3, 5, 7],
    'model__learning_rate': [0.05, 0.1]
}

scores_xgb = cross_val_score(build_pipeline(XGBRegressor(random_state=42, verbosity=0)), x_train, y_train, cv=5, scoring='r2')
print(scores_xgb)
print(f"XGBoost CV R2: {scores_xgb.mean():.4f}")

xgb_grid = GridSearchCV(
    build_pipeline(XGBRegressor(random_state=42, verbosity=0)),
    xgb_param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1
)

xgb_grid.fit(x_train, y_train)
final_pipeline = xgb_grid.best_estimator_
print(f"\nXGBoost Best Params: {xgb_grid.best_params_}")
print(f"XGBoost Best Score: {xgb_grid.best_score_:.4f}")

# ============================================================
# 10. CROSS-VAL PREDICTIONS FOR RESIDUAL ANALYSIS (Phase 2)
# cross_val_predict give leak-free predictions for each row
# when it was in the validation fold.
# Used for residual analysis — not for final evaluation.
# Final evaluation always uses the holdout set.
# ============================================================



y_pred_cv = cross_val_predict(final_pipeline,x_train,y_train, cv=5)

y_train_actual = np.expm1(y_train)
y_pred_actual = np.expm1(y_pred_cv)

mae_cv = mean_absolute_error(y_train_actual,y_pred_actual)
rmse_cv = root_mean_squared_error(y_train_actual,y_pred_actual)
r2_cv = r2_score(y_train,y_pred_cv)

print(f"\n—— CV Performance (training set) ——")
print(f"MAE: ${mae_cv:,.0f}")
print(f"RMSE: ${rmse_cv:,.0f}")
print(f"R²:   {r2_cv:.4f}")


# ============================================================
# 11. FINAL EVALUATION ON HOLDOUT SET (Phase 2)
# This is the ONLY time we touch the holdout set.
# Fit the final pipeline on ALL training data first,
# then predict on holdout.
# CV R² ≈ Holdout R² → model generalizes well
# CV R² >> Holdout R² → model overfit to training data
# ============================================================


final_pipeline.fit(x_train, y_train)
y_test_pred = final_pipeline.predict(x_test)

y_test_actual = np.expm1(y_test)
y_test_pred_actual = np.expm1(y_test_pred)

mae_t = mean_absolute_error(y_test_actual,y_test_pred_actual)
rmse_t = root_mean_squared_error(y_test_actual,y_test_pred_actual)
r2_t = r2_score(y_test,y_test_pred)

print(f"\n—— Test Performance (final honest score) ——")
print(f"MAE: ${mae_t:,.0f}")
print(f"RMSE: ${rmse_t:,.0f}")
print(f"R²:   {r2_t:.4f}")

print(f"\nMAE vs RMSE gap: ${rmse_t - mae_t:,.0f}")
print("(Large gap = some very large individual errors)")


# ============================================================
# 12. SHAP EXPLANATION (Phase 4)
# SHAP explains WHY the model made each prediction.
# Without SHAP, you can only say which features matter globally.
# With SHAP, you can say exactly why THIS house got THIS price.
# This is critical for real-world trust and auditability.
# ============================================================

# Extract the two steps from the fitted pipeline
# We need them separately because SHAP works directly
# on the model — not the full pipeline
preprocess_step = final_pipeline.named_steps['preprocessor']
xgb_model = final_pipeline.named_steps['model']

# Transform testing data using the fitted preprocessor
# This gives us the actual numerical array the XGBoost
# model received — SHAP needs this exact input
x_test_transformed = preprocess_step.transform(x_test)

# Get feature names after preprocessing
# Numerical columns keep their original names
# Categorical columns become e.g. "Neighborhood_OldTown"
cat_feature_names = (
    preprocess_step
    .named_transformers_['cat']['encoder']
    .get_feature_names_out(categorical_col)
    .tolist()
)
all_feature_names = numerical_col + cat_feature_names

# TreeExplainer is optimized specifically for tree-based models
# It computes exact SHAP values efficiently
# For non-tree models you would use shap.Explainer instead
explainer   = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(x_test_transformed)

# ── PLOT 1: Global Feature Importance ─────────────────────
# Bar length = average absolute SHAP value across all houses
# Absolute means direction (up/down) is ignored — only
# magnitude matters here.
# Longer bar = feature moves predictions more on average.
plt.figure()
shap.summary_plot(
    shap_values,
    x_test_transformed,
    feature_names = all_feature_names,
    plot_type = 'bar',
    max_display = 15,
    show = False
)
plt.title("Global Feature Importance (SHAP)")
plt.tight_layout()
plt.savefig("plots/shap_global.png", bbox_inches='tight')
plt.show()

# ── PLOT 2: SHAP Dot Plot ──────────────────────────────────
# Each dot = one house in holdout set
# Position on x-axis = how much it pushed price up or down
# Color = feature value (red = high value, blue = low value)
# Red dots on right = high value pushes price UP (e.g. TotalSF)
# Blue dots on left = low value pushes price DOWN
# If you see red dots on LEFT something is wrong with your data

plt.figure()
shap.summary_plot(
    shap_values,
    x_test_transformed,
    feature_names = all_feature_names,
    max_display = 15,
    show = False
)
plt.title("SHAP Feature Impact (Dot Plot)")
plt.tight_layout()
plt.savefig("plots/shap_dot.png", bbox_inches='tight')
plt.show()

# ── PLOT 3: Individual Prediction Explanation ──────────────
# Pick one house and explain exactly why the model
# predicted that specific price.
# This is what you show a homeowner who disputes the price.
house_index = 0
predicted_price = np.expm1(y_test_pred[house_index])
actual_price = np.expm1(y_test.iloc[house_index])

print(f"\n—— Individual Prediction Explanation ——")
print(f"Actual Price: ${actual_price:,.0f}")
print(f"Predicted Price: {predicted_price:,.0f}")
print(f"Difference: {abs(actual_price - predicted_price):,.0f}")

shap.waterfall_plot(
    shap.Explanation(
        values = shap_values[house_index],
        base_values = explainer.expected_value,
        data = x_test_transformed[house_index],
        feature_names = all_feature_names
    ),
    max_display = 12,
    show = False
)
plt.title(f"Why did the model predict ${predicted_price:,.0f}?")
plt.tight_layout()
plt.savefig("plots/shap_individual.png", bbox_inches='tight')
plt.show()




# results['XGBoost'] = {
#     "MAE": mean_absolute_error(np.expm1(y), np.expm1(y_pred_xgb)),
#     "RMSE": np.sqrt(mean_squared_error(np.expm1(y), np.expm1(y_pred_xgb))),
#     "R2 Score": r2_score(y, y_pred_xgb)
# }

#  # Print results
# print("\n===== MODEL COMPARISON =====")
# for model_name, metrics in results.items():
#     print(f"\n{model_name}:")
#     print(f"  MAE: ${metrics['MAE']:,.2f}")
#     print(f"  RMSE: ${metrics['RMSE']:,.2f}")
#     print(f"  R2: {metrics['R2 Score']:.4f}")

# ============================================================
# 13. VISUALIZATION
# ============================================================
# Histogram plot for SalePrice column (Checks number of houses within available price range)
plt.figure(figsize=(8,5))
sns.histplot(train['SalePrice'], kde=True, color='black')
plt.title('Distribution of Sale Price after Log Transformation')
plt.savefig("plots/Distribution_of_SalePrice_after_Log_Transformation.png", bbox_inches="tight")
plt.show()

# Heatmap to show correlation between columns
plt.figure(figsize=(12,8))
sns.heatmap(x_train.select_dtypes(include=[np.number]).corr(), cmap='coolwarm')
plt.title("Correlation Heat Map")
plt.savefig("plots/Correlation_Heat_Map.png", bbox_inches="tight")
plt.show()

# Actual vs. Predicted Prices
plt.figure(figsize=(6,6))
plt.scatter(np.expm1(y_test),np.expm1(y_test_pred) , alpha=0.5)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices(XGBoost)")

# Perfect prediction line
plt.plot([np.expm1(y_test).min(), np.expm1(y_test).max()], [np.expm1(y_test).min(), np.expm1(y_test).max()], color='red')
plt.savefig("plots/Actual_vs_Predicted_Prices(XGBoost).png", bbox_inches="tight")
plt.show()

# Important features plot
# importance = xgb_best_model.feature_importances_

# Residual analysis plots
residuals = np.expm1(y_test).values - np.expm1(y_test_pred)

fig,axes = plt.subplots(1,3,figsize=(16,5))
fig.suptitle("Residual Analysis — Holdout Set", fontsize=14)

# Actual vs. Predicted
axes[0].scatter(
    np.expm1(y_test),
    np.expm1(y_test_pred),
    alpha=0.4, color='steelblue'
)
axes[0].plot(
    [np.expm1(y_test).min(), np.expm1(y_test).max()],
    [np.expm1(y_test).min(), np.expm1(y_test).max()],
    color='red', linewidth=2
)
axes[0].set_xlabel("Actual Price ($)")
axes[0].set_ylabel("Predicted Price ($)")
axes[0].set_title("Actual vs Predicted")

# Residuals vs. Predicted
# Should look like random horizontal scatter around zero
# Funnel shape = model less accurate for expensive houses
axes[1].scatter(
    np.expm1(y_test_pred),
    residuals,
    alpha=0.4, color='steelblue'
)
axes[1].axhline(y=0, color='red', linewidth=2)
axes[1].set_xlabel("Predicted Price ($)")
axes[1].set_ylabel("Residual ($)")
axes[1].set_title("Residuals vs Predicted")

# Residual distribution
# Should be bell curve centred at zero
# Skewed = model systematically over or underestimates
axes[2].hist(residuals, bins=40, color='steelblue', edgecolor='white')
axes[2].axvline(x=0, color='red', linewidth=2)
axes[2].set_xlabel("Residual ($)")
axes[2].set_ylabel("Count")
axes[2].set_title("Residual Distribution")

plt.tight_layout()
plt.savefig("plots/residual_analysis.png", bbox_inches='tight')
plt.show()

print(f"\nMean residual:    ${residuals.mean():,.0f}  (close to 0 = unbiased)")
print(f"Std of residuals: ${residuals.std():,.0f}  (spread of errors)")

# ============================================================
# 14. SAVE MODEL & METADATA (Phase 5)
# Save the full pipeline — not just the XGBoost model.
# The pipeline contains the fitted imputer, scaler, and
# encoder — all the learned statistics from training.
# Without them, you cannot preprocess new inputs correctly.
# Joblib is used instead of pickle because it is faster
# and more memory efficient for large numpy arrays.
# ============================================================

os.makedirs('models', exist_ok=True)

# Full pipeline — used by Streamlit app for predictions
joblib.dump(final_pipeline, 'models/house_price_pipeline.pkl')

# Column lists — used by Streamlit to validate input shape
joblib.dump(list(x.columns),  'models/feature_columns.pkl')
joblib.dump(numerical_col,   'models/numerical_cols.pkl')
joblib.dump(categorical_col, 'models/categorical_cols.pkl')

print("\nModel and metadata saved to models/")
print(f"  house_price_pipeline.pkl")
print(f"  feature_columns.pkl")
print(f"  numerical_cols.pkl")
print(f"  categorical_cols.pkl")





# features = x.columns
# feat_imp = pd.DataFrame({
#     'feature': features,
#     'importance': importance
# })
#
# feat_imp = feat_imp.sort_values(by='importance', ascending=False)
#
# top_features = feat_imp.head(10)
#
# plt.figure()
# plt.barh(top_features['feature'], top_features['importance'])
# plt.gca().invert_yaxis()
# plt.title("Top 10 Important Features")
# plt.savefig("plots/top_features.png", bbox_inches="tight")
# plt.show()
#
# # Scatterplot for showing relationship between SalePrice and top columns which had high correlation with SalePrice
# fig, axes = plt.subplots(2, 3, figsize=(15,10))
# axes = axes.flatten()
# top_6_features = feat_imp.head(6)
# for i, feature in enumerate(top_6_features['feature']):
#     axes[i].scatter(train[feature], train['SalePrice'], alpha=0.5)
#     axes[i].set_xlabel(feature)
#     axes[i].set_ylabel('SalePrice')
#     axes[i].set_title(f'{feature} vs SalePrice')
# plt.tight_layout()
# plt.savefig("plots/relationship_of_top_6_features_with_SalePrice.png", bbox_inches="tight")
# plt.show()


# ============================================================
# 15. KAGGLE SUBMISSION
# ============================================================

test_ids = test['Id']
test = test.drop('Id', axis=1)
# Predict using XGBoost
test_predictions = np.expm1(final_pipeline.predict(test))

# Submission file
submission = pd.DataFrame({
    'Id': test_ids,
    'SalePrice': test_predictions
})
submission.to_csv('submission.csv', index=False)
print(submission.head())