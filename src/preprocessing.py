import pandas as pd
from sklearn.preprocessing import (
    LabelEncoder,
    StandardScaler
)
# =========================
# LOAD DATA
def load_data(path):
    df = pd.read_csv(path)
    return df
df = load_data("train.csv")

print(df.head())
# HANDLE MISSING VALUES
# =========================

def handle_missing_values(df):

    # Numerical columns
    num_cols = df.select_dtypes(
        include=['int64', 'float64']
    ).columns

    # Categorical columns
    cat_cols = df.select_dtypes(
        include=['object']
    ).columns

    # Fill numerical missing values
    for col in num_cols:

        df[col] = df[col].fillna(
            df[col].median()
        )

    # Fill categorical missing values
    for col in cat_cols:

        df[col] = df[col].fillna(
            df[col].mode()[0]
        )

    return df

# ENCODE FEATURES

def encode_features(df):

    cat_cols = df.select_dtypes(
        include=['object']
    ).columns

    encoder = LabelEncoder()

    for col in cat_cols:

        df[col] = encoder.fit_transform(
            df[col]
        )

    return df
# SCALE FEATURES
def scale_features(X):

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    return X_scaled