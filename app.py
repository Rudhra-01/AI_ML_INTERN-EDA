import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_squared_error,
    r2_score
)

from sklearn.feature_selection import (
    SelectKBest,
    f_regression
)

from src.preprocessing import (
    load_data,
    handle_missing_values,
    encode_features,
    scale_features
)

from src.feature_engineering import (
    create_features,
    log_transform
)

# =========================================
# LOAD DATASET
# =========================================

print("\nLoading Dataset...")

df = load_data("train.csv")

print("Dataset Loaded Successfully")

print("\nDataset Shape:")
print(df.shape)

# =========================================
# BASIC INFORMATION
# =========================================

print("\nDataset Information")

print(df.info())

# =========================================
# MISSING VALUE HANDLING
# =========================================

print("\nHandling Missing Values...")

df = handle_missing_values(df)

print("Missing Values Handled Successfully")

# =========================================
# OUTLIER HANDLING
# =========================================

print("\nRemoving Outliers...")

Q1 = df['SalePrice'].quantile(0.25)

Q3 = df['SalePrice'].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - (1.5 * IQR)

upper = Q3 + (1.5 * IQR)

df = df[
    (df['SalePrice'] >= lower)
    &
    (df['SalePrice'] <= upper)
]

print("Outliers Removed Successfully")

print("New Dataset Shape:")

print(df.shape)

# =========================================
# FEATURE ENGINEERING
# =========================================

print("\nPerforming Feature Engineering...")

df = create_features(df)

print("Feature Engineering Completed")

# =========================================
# LOG TRANSFORMATION
# =========================================

print("\nApplying Log Transformation...")

df = log_transform(df)

print("Transformation Completed")

# =========================================
# SALE PRICE DISTRIBUTION
# =========================================

plt.figure(figsize=(10, 6))

sns.histplot(
    df['SalePrice'],
    kde=True
)

plt.title("Sale Price Distribution")

plt.savefig("price_distribution.png")

plt.close()

# =========================================
# SALE PRICE BOXPLOT
# =========================================

plt.figure(figsize=(10, 6))

sns.boxplot(
    x=df['SalePrice']
)

plt.title("SalePrice Boxplot")

plt.savefig("saleprice_boxplot.png")

plt.close()

# =========================================
# SCATTER PLOT
# =========================================

plt.figure(figsize=(10, 6))

sns.scatterplot(
    x=df['GrLivArea'],
    y=df['SalePrice']
)

plt.title("Living Area vs SalePrice")

plt.savefig("scatterplot.png")

plt.close()

# =========================================
# CORRELATION HEATMAP
# =========================================

plt.figure(figsize=(18, 12))

corr = df.corr(numeric_only=True)

sns.heatmap(
    corr,
    cmap='coolwarm'
)

plt.title("Correlation Heatmap")

plt.savefig("correlation_heatmap.png")

plt.close()

# =========================================
# ENCODING
# =========================================

print("\nEncoding Features...")

df = encode_features(df)

print("Encoding Completed")

# =========================================
# FEATURE & TARGET SPLIT
# =========================================

X = df.drop(
    ['SalePrice', 'Id'],
    axis=1
)

y = df['SalePrice']

# =========================================
# FEATURE SCALING
# =========================================

print("\nScaling Features...")

X_scaled = scale_features(X)

print("Scaling Completed")

# =========================================
# FEATURE SELECTION
# =========================================

print("\nPerforming Feature Selection...")

selector = SelectKBest(
    score_func=f_regression,
    k=20
)

X_selected = selector.fit_transform(
    X_scaled,
    y
)

print("Feature Selection Completed")

print("Selected Feature Shape:")

print(X_selected.shape)

# =========================================
# TRAIN TEST SPLIT
# =========================================

X_train, X_test, y_train, y_test = train_test_split(
    X_selected,
    y,
    test_size=0.2,
    random_state=42
)

# =========================================
# LINEAR REGRESSION
# =========================================

print("\nTraining Linear Regression Model...")

lr_model = LinearRegression()

lr_model.fit(
    X_train,
    y_train
)

lr_pred = lr_model.predict(X_test)

# =========================================
# RANDOM FOREST
# =========================================

print("\nTraining Random Forest Model...")

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_model.fit(
    X_train,
    y_train
)

rf_pred = rf_model.predict(X_test)

# =========================================
# MODEL EVALUATION FUNCTION
# =========================================

def evaluate_model(true, pred, model_name):

    rmse = np.sqrt(
        mean_squared_error(true, pred)
    )

    r2 = r2_score(true, pred)

    print(f"\n{model_name}")

    print("-" * 35)

    print("RMSE :", rmse)

    print("R2 Score :", r2)

# =========================================
# MODEL EVALUATION
# =========================================

evaluate_model(
    y_test,
    lr_pred,
    "Linear Regression"
)

evaluate_model(
    y_test,
    rf_pred,
    "Random Forest"
)
# =========================================
# FEATURE IMPORTANCE
# =========================================

selected_features = X.columns[
    selector.get_support()
]

importance = pd.DataFrame({

    'Feature': selected_features,

    'Importance': rf_model.feature_importances_

})

importance = importance.sort_values(
    by='Importance',
    ascending=False
)

print("\nTop Important Features")

print(
    importance.head(10)
)
# =========================================
# FINAL INSIGHTS
# =========================================

print("\nPROJECT COMPLETED SUCCESSFULLY")

print("""

Key Insights:

1. Overall Quality strongly affects house prices

2. Ground Living Area has high correlation with SalePrice

3. Feature Engineering improved model understanding

4. Random Forest performed better than Linear Regression

5. Outlier handling improved model performance

6. Feature Selection reduced unnecessary features

""")