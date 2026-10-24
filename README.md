# Ames Housing Price Prediction

Complete end-to-end machine learning regression workflow on the **Ames Housing** dataset.

Predict house sale prices using careful preprocessing, exploratory analysis, and multiple models (Linear Regression, SVR, KNN) with hyperparameter tuning.

---

## Project Overview

| Item | Details |
|------|---------|
| **Dataset** | Ames Housing (1,460 houses, 80 features) |
| **Target** | `SalePrice` |
| **Task** | Regression |
| **Best Model** | Linear Regression (Test R² ≈ 0.92, RMSE ≈ $24.8k) |
| **Key Techniques** | Custom imputation, ColumnTransformer pipelines, log-target transform, GridSearchCV, Lasso feature selection |

---

## Workflow

1. **Dataset Understanding** – column renaming, feature grouping, missing-value analysis  
2. **Exploratory Data Analysis (EDA)** – distributions, correlations, neighborhood effects  
3. **Data Insights** – quality, size, location, and amenities as main price drivers  
4. **Preprocessing** – pipelines for numerical / ordinal / nominal features + target transform  
5. **Model Development** – Linear Regression, SVR, KNN  
6. **Hyperparameter Tuning** – GridSearchCV (5-fold CV)  
7. **Evaluation & Comparison** – MAE, RMSE, R² on Train / Validation / Test  
8. **Feature Importance** – coefficients & permutation importance  
9. **Final Model Selection**

---

## Results Summary

| Model              | Val RMSE | Val R² | Test RMSE | Test R² | R² Gap |
|--------------------|----------|--------|-----------|---------|--------|
| **Linear Regression** | **24,140** | **0.910** | **24,768** | **0.920** | 0.035 |
| SVR (default)      | 26,114   | 0.895  | 29,658    | 0.885   | 0.043  |
| SVR (tuned)        | 31,305   | 0.848  | 36,634    | 0.825   | 0.012  |
| KNN (tuned)        | 35,898   | 0.801  | 35,999    | 0.831   | 0.092  |
| KNN (default)      | 36,060   | 0.799  | 36,565    | 0.826   | 0.046  |

**Final selected model: Linear Regression**  
(lowest validation RMSE, best test performance, small generalization gap, fully interpretable)

---

## Key Findings

- **Overall quality**, **above-grade living area**, **basement size**, and **neighborhood** are the strongest predictors of sale price.
- Many “missing” values actually mean the feature does not exist (no pool, no alley, no fireplace, etc.).
- A well-designed preprocessing pipeline allows a simple Linear Regression to achieve strong performance (Test R² ≈ 0.92).
- Hyperparameter tuning of SVR and KNN brought limited gains compared to the linear baseline.

---

## Repository Structure
