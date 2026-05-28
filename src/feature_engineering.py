import numpy as np
# CREATE FEATURES
def create_features(df):

    # House Age
    df['HouseAge'] = (
        df['YrSold'] - df['YearBuilt']
    )

    # Total Square Feet
    df['TotalSF'] = (
        df['TotalBsmtSF']
        + df['1stFlrSF']
        + df['2ndFlrSF']
    )

    # Total Bathrooms
    df['TotalBath'] = (
        df['FullBath']
        + (0.5 * df['HalfBath'])
        + df['BsmtFullBath']
        + (0.5 * df['BsmtHalfBath'])
    )

    return df
# LOG TRANSFORMATION
def log_transform(df):
    df['SalePrice'] = np.log1p(
        df['SalePrice']
    )
    return df