# ✈️ BANGLADESH-FLIGHT-PRICE-ANALYSIS

This project focuses on an in-depth **Exploratory Data Analysis (EDA)** and **Predictive Modeling** to analyze and predict **flight prices in Bangladesh** using advanced feature selection and machine learning techniques.

---

## 🔍 Project Overview

- Performed **data cleaning and preprocessing** to handle missing values, inconsistent formats, and irrelevant features.
- Conducted **Exploratory Data Analysis (EDA)** to understand the distribution, correlation, and patterns in the dataset.
- Applied **feature selection techniques** to identify the most influential factors affecting flight prices.

---

## 🧠 Feature Selection Techniques Used

- **F-Regression (`f_regression`)**: Measures the strength of **linear relationships** between independent features and the target variable (`Total Fare (BDT)`).
- **Mutual Information Regression (`mutual_info_regression`)**: Captures **non-linear dependencies** between features and the target.
- **Random Forest Feature Importance**: Tree-based model to evaluate each feature's contribution.
- **XGBoost Feature Importance**: Gradient boosting-based method to determine feature significance.
- Feature scores were **normalized using Min-Max Scaling**, and an **average score** was computed to select the **Top 11 Most Influential Features**.
- A **bar plot** was used to visualize the top features based on their average importance scores.

---

## 📈 Predictive Modeling

Three powerful regression models were implemented to predict flight fares:

- **Artificial Neural Network (ANN)** – R² Score: **0.9969**
- **Random Forest Regressor (RF)** – R² Score: **0.9999**
- **XGBoost Regressor (XGB)** – R² Score: **0.9995**

These models demonstrated **high accuracy and strong generalization** capabilities for predicting flight prices.

---

## 📊 Visualizations

- **Histograms** for:
  - Base Fare (BDT)
  - Tax & Surcharge (BDT)
  - Total Fare (BDT)
- **Bar chart** for:
  - Average Feature Importance Scores (Top 11 Features)
- **Model evaluation metrics**:
  - R-squared (R²)
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)

---

## 💻 Dependencies

```python
# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Data preprocessing and feature selection
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.feature_selection import f_regression, mutual_info_regression

# Machine Learning Models
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Deep Learning (ANN)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Model Evaluation & Data Splitting
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
