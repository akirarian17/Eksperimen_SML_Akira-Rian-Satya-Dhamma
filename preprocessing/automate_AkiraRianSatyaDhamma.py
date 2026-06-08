import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


INPUT_FILE = "weatherAUS_raw.csv"
OUTPUT_FILE = "./preprocessing/weatherAUS_uluru_preprocessed.csv"

def load_data(filepath):
    return pd.read_csv(filepath)


def filter_location(df):
    return df[df["Location"] == "Uluru"].copy()


def remove_duplicates(df):
    before = len(df)

    df = df.drop_duplicates()

    after = len(df)

    print(f"Duplicates removed: {before - after}")

    return df


def remove_high_missing_columns(df, threshold=50):
    missing_percentage = df.isnull().mean() * 100

    cols_to_drop = missing_percentage[
        missing_percentage > threshold
    ].index.tolist()

    print("\nColumns dropped (>50% missing):")
    print(cols_to_drop)

    df = df.drop(columns=cols_to_drop)

    return df


def convert_date_features(df):
    df["Date"] = pd.to_datetime(df["Date"])

    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day

    df = df.drop(columns=["Date"])

    return df


def handle_missing_values(df):
    numerical_columns = df.select_dtypes(
        include=np.number
    ).columns

    categorical_columns = df.select_dtypes(
        exclude=np.number
    ).columns

    for col in numerical_columns:
        median_value = df[col].median()
        df[col] = df[col].fillna(median_value)

    for col in categorical_columns:
        mode_value = df[col].mode()[0]
        df[col] = df[col].fillna(mode_value)

    return df


def encode_categorical_features(df):
    categorical_columns = df.select_dtypes(
        include="object"
    ).columns

    encoders = {}

    for col in categorical_columns:
        encoder = LabelEncoder()

        df[col] = encoder.fit_transform(df[col])

        encoders[col] = encoder

    return df, encoders


def detect_outliers_iqr(df):
    numerical_columns = df.select_dtypes(
        include=np.number
    ).columns

    for col in numerical_columns:

        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)

        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        df[col] = np.where(
            df[col] < lower_bound,
            lower_bound,
            df[col]
        )

        df[col] = np.where(
            df[col] > upper_bound,
            upper_bound,
            df[col]
        )

    return df


def standardize_features(df):
    scaler = StandardScaler()

    target_column = "RainTomorrow"

    feature_columns = [
        col for col in df.columns
        if col != target_column
    ]

    df[feature_columns] = scaler.fit_transform(
        df[feature_columns]
    )

    return df, scaler


def preprocessing_pipeline(filepath):
    print("Loading data...")
    df = load_data(filepath)

    print("Filtering Uluru...")
    df = filter_location(df)

    print("Removing duplicates...")
    df = remove_duplicates(df)

    print("Removing high-missing columns...")
    df = remove_high_missing_columns(df)

    print("Creating date features...")
    df = convert_date_features(df)

    print("Handling missing values...")
    df = handle_missing_values(df)

    print("Encoding categorical features...")
    df, encoders = encode_categorical_features(df)

    print("Handling outliers...")
    df = detect_outliers_iqr(df)

    print("Standardizing features...")
    df, scaler = standardize_features(df)

    print("\nFinal Shape:")
    print(df.shape)

    return df


def main():
    processed_df = preprocessing_pipeline(INPUT_FILE)

    processed_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"\nPreprocessed dataset saved to: {OUTPUT_FILE}"
    )

main()