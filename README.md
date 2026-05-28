# House Price EDA & Prediction Project

## Project Overview

This project performs complete Exploratory Data Analysis (EDA), data preprocessing, feature engineering, feature selection, and machine learning on the Kaggle House Prices dataset.

The project demonstrates a professional end-to-end data science workflow including:

* Data Cleaning
* Missing Value Handling
* Outlier Detection
* Feature Engineering
* Encoding
* Feature Scaling
* Feature Selection
* Data Visualization
* Machine Learning Model Building
* Model Evaluation

---

# Dataset Information

Dataset: House Prices - Advanced Regression Techniques

Source:
[https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)

The dataset contains multiple numerical and categorical features related to residential homes and predicts house sale prices.

---

# Technologies Used

## Programming Language

* Python

## Libraries

* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

---

# Project Structure

```bash
AI-ML-Intern-EDA/
│
├── data/
│   ├── train.csv
│   ├── test.csv
│
├── notebooks/
│   ├── 01_Data_Cleaning.ipynb
│   ├── 02_EDA_Visualization.ipynb
│   ├── 03_Feature_Engineering.ipynb
│   ├── 04_Model_Building.ipynb
│
├── images/
│   ├── correlation_heatmap.png
│   ├── feature_importance.png
│   ├── scatterplot.png
│
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
```

---

# EDA & Preprocessing Steps

## 1. Data Cleaning

* Loaded dataset using Pandas
* Checked dataset structure and missing values
* Handled null values using median and mode imputation

## 2. Missing Value Handling

### Numerical Columns

* Median Imputation

### Categorical Columns

* Mode Imputation

## 3. Outlier Handling

* Used IQR Method
* Removed extreme SalePrice outliers

## 4. Feature Engineering

Created new features:

* HouseAge
* TotalSF
* TotalBath

## 5. Encoding

* Label Encoding applied to categorical variables

## 6. Feature Scaling

* StandardScaler used for normalization

## 7. Feature Selection

* SelectKBest
* f_regression

## 8. Data Visualization

Visualizations created:

* Sale Price Distribution
* Correlation Heatmap
* Scatter Plot
* Feature Importance Plot
* Boxplot

---

# Machine Learning Models

## Models Used

* Linear Regression
* Random Forest Regressor

## Evaluation Metrics

* RMSE (Root Mean Squared Error)
* R² Score

---

# Key Insights

* Overall Quality strongly affects house prices
* Living Area has high correlation with SalePrice
* Feature engineering improved model performance
* Random Forest performed better than Linear Regression
* Outlier removal improved prediction quality
* Feature selection reduced unnecessary features

---

# Class Balancing

Class balancing techniques were not applied because the dataset is regression-based rather than classification-based.

---

# How to Run Project

## 1. Create Virtual Environment

```bash
python -m venv venv
```
## 2. Activate Environment

### Windows

```bash
venv\Scripts\activate
```
## 3. Install Requirements

```bash
pip install -r requirements.txt
```
## 4. Run Project

```bash
python app.py
```
---

# Future Improvements

* Add XGBoost Regressor
* Deploy using Streamlit
* Add interactive dashboard
* Perform hyperparameter tuning
* Add SHAP explainability

---

# Author
Rudhra Roopini
---

# Conclusion

This project demonstrates a complete end-to-end machine learning workflow including data preprocessing, exploratory data analysis, feature engineering, feature selection, visualization, and regression model building using Python.
It serves as a strong beginner-to-intermediate level data science portfolio project suitable for internships and GitHub showcase.
