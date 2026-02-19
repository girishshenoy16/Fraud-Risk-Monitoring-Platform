import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def run_eda(
        input_path="data/raw/creditcard.csv",
        output_path="outputs/plots/"
):
    print("Running EDA...")

    df = pd.read_csv(input_path)

    print("Dataset Shape:", df.shape)
    print("\nMissing Values:\n", df.isnull().sum().sum())
    print("\nClass Distribution:\n", df["Class"].value_counts())

    os.makedirs(output_path, exist_ok=True)

    # -------------------------
    # 1. Class Imbalance Plot
    # -------------------------
    plt.figure()
    sns.countplot(x="Class", data=df)
    plt.title("Class Distribution (Fraud vs Non-Fraud)")
    plt.savefig(output_path + "class_distribution.png")
    plt.close()

    # -------------------------
    # 2. Amount Distribution
    # -------------------------
    plt.figure()
    sns.histplot(df[df["Class"] == 0]["Amount"], bins=50, label="Normal", kde=True)
    sns.histplot(df[df["Class"] == 1]["Amount"], bins=50, label="Fraud", kde=True, color="red")
    plt.legend()
    plt.title("Transaction Amount Distribution")
    plt.savefig(output_path + "amount_distribution.png")
    plt.close()

    # -------------------------
    # 3. Correlation Heatmap
    # -------------------------
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), cmap="coolwarm", linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.savefig(output_path + "correlation_heatmap.png")
    plt.close()

    print("EDA completed. Plots saved in outputs/plots/")


if __name__ == "__main__":
    run_eda()