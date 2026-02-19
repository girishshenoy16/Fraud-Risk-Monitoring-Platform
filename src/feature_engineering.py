import pandas as pd
import numpy as np
import os


def engineer_features(
    input_path="data/processed/processed_creditcard.csv",
    output_path="data/processed/featured_creditcard.csv"
):
    print("Running Feature Engineering...")

    df = pd.read_csv(input_path)

    # ------------------------------------------------
    # Ensure Required Columns Exist
    # ------------------------------------------------
    if "Amount_original" not in df.columns:
        raise ValueError("Amount_original column not found. Check preprocessing step.")

    if "Time_original" not in df.columns:
        raise ValueError("Time_original column not found. Check preprocessing step.")

    # Log transform amount
    df["log_amount"] = np.log1p(np.abs(df["Amount_original"]))

    # Ratio feature
    df["amount_time_ratio"] = (
            df["Amount_original"] / (df["Time_original"] + 1)
    )

    # High amount flag
    df["high_amount_flag"] = (
            df["Amount_original"] >
            df["Amount_original"].quantile(0.95)
    ).astype(int)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print("Feature engineering completed.")
    print("New shape:", df.shape)


if __name__ == "__main__":
    engineer_features()