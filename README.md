# 🧠 Automobile Price Prediction - Exploratory Data Analysis (EDA) + Feature Interpretation

This project performs an end-to-end Exploratory Data Analysis (EDA), preprocessing, dimensionality reduction, and SHAP-based feature importance analysis on the [1985 UCI Automobile Dataset](https://archive.ics.uci.edu/ml/datasets/automobile). The goal is to uncover the most influential factors affecting **automobile prices** and visualize their importance through machine learning models and explainability tools.

---

## 📌 Table of Contents

- [📊 Dataset Overview](#-dataset-overview)
- [🧼 Data Cleaning & Preprocessing](#-data-cleaning--preprocessing)
- [📈 Exploratory Data Analysis (EDA)](#-exploratory-data-analysis-eda)
- [⚙️ Feature Engineering & PCA](#️-feature-engineering--pca)
- [🌲 Random Forest + SHAP Interpretability](#-random-forest--shap-interpretability)
- [📌 Key Insights](#-key-insights)
- [📎 Tools Used](#-tools-used)
- [🧠 What I Learned](#-what-i-learned)
- [📁 Files](#-files)

---

## 📊 Dataset Overview

- **Source**: UCI ML Repository (1985 Automobile Dataset)
- **Records**: ~205 entries (after cleaning)
- **Columns**: 26 features including both numeric and categorical attributes
- **Target Variable**: `price` of the vehicle

---

## 🧼 Data Cleaning & Preprocessing

- Handled **missing values**:
  - Numeric missing values imputed with **mean**
  - Categorical values dropped or encoded
- Converted string representations of numbers to float (`?` replaced)
- Removed outliers in the `price` and `horsepower` distributions
- Verified datatypes — all inputs cast to numeric (float)

---

## 📈 Exploratory Data Analysis (EDA)

### 🔹 Univariate Analysis
- Histogram & boxplots for key features: `price`, `engine-size`, `horsepower`, `curb-weight`, etc.
- Price distribution was **right-skewed**, so log transformation was tested.
- Identified top 5 most expensive brands.

### 🔹 Bivariate/Multivariate Analysis
- **Correlation heatmaps** to detect multicollinearity.
- Scatterplots for `engine-size` vs `price`, `horsepower` vs `price`.
- **Brand vs average price** barplots.
- Category breakdowns:
  - `fuel-type`, `body-style`, `drive-wheels` vs price
  - Found that **RWD (rear-wheel drive)** and **diesel** vehicles tend to be more expensive.

---

## ⚙️ Feature Engineering & PCA

- One-hot encoded all categorical columns (`get_dummies`)
- Scaled numeric features using `StandardScaler`
- Performed **Principal Component Analysis (PCA)** to reduce dimensionality:
  - Retained components that explained >95% of variance
  - Used `pca_1`, `pca_2` as additional features in the model

---

## 🌲 Random Forest + SHAP Interpretability

- Trained a **RandomForestRegressor** on the full feature set (raw + PCA)
- Used **SHAP (SHapley Additive Explanations)** to understand the black-box predictions
- Visualized feature contributions using:
  - **SHAP beeswarm plot**
  - Ranked features based on average absolute SHAP values

---

## 📌 Key Insights

- `pca_1` (principal component from original numeric data) was the **most powerful predictor**, capturing multiple interacting features.
- Raw features like `highway-mpg`, `width`, and `horsepower` still had **high predictive value**, suggesting that PCA did not fully absorb their individual contributions.
- `drive-wheels_rwd` and `body-style_hatchback` had noticeable effects, confirming domain intuition.
- Features with near-zero SHAP values were candidates for removal or de-prioritization.

---

## 📎 Tools Used

| Task | Library/Tool |
|------|--------------|
| EDA & Visualization | `pandas`, `matplotlib`, `seaborn` |
| Preprocessing       | `sklearn.preprocessing` |
| Modeling            | `RandomForestRegressor` |
| Dimensionality Reduction | `sklearn.decomposition.PCA` |
| Interpretability    | `shap` |
| Notebook Runtime    | Google Colab / Jupyter |

---

## 🧠 What I Learned

- Conducting deep EDA before modeling is **non-negotiable** for insight discovery.
- PCA can reduce noise but doesn’t always remove the value of key raw features.
- SHAP enables **trustworthy interpretation** of complex models like Random Forests.
- Visual storytelling with data (via SHAP plots) reveals much more than metrics alone.

---

## ✅ Next Steps (Optional)

- Try `XGBoost` or `LightGBM` + SHAP
- Build a regression pipeline using `Pipeline()` in `sklearn`
- Try Lasso/Ridge regression to verify feature regularization
- Deploy this as a **streamlit-based interactive app**

---



