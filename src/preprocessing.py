import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib


def preprocess_data(
        input_path="data/raw/creditcard.csv",
        output_path="data/processed/processed_creditcard.csv",
        scaler_path="models/scaler.pkl"
):
    print("Loading dataset...")
    df = pd.read_csv(input_path)

    print("Initial Shape:", df.shape)

    if df.isnull().sum().sum() != 0:
        raise ValueError("Dataset contains missing values!")

    # -----------------------------------------
    # Preserve Original Columns
    # -----------------------------------------
    df["Time_original"] = df["Time"]
    df["Amount_original"] = df["Amount"]

    # -----------------------------------------
    # Scaling
    # -----------------------------------------
    scaler = StandardScaler()

    df["Time_scaled"] = scaler.fit_transform(df[["Time"]])
    df["Amount_scaled"] = scaler.fit_transform(df[["Amount"]])

    # -----------------------------------------
    # Drop original columns used for modeling
    # (but keep originals for dashboard)
    # -----------------------------------------
    df.drop(columns=["Time", "Amount"], inplace=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)

    df.to_csv(output_path, index=False)
    joblib.dump(scaler, scaler_path)

    print("Preprocessing complete.")
    print("Processed file saved.")


if __name__ == "__main__":
    preprocess_data()