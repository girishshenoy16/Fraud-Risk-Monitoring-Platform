import joblib
import matplotlib.pyplot as plt
import pandas as pd
import os


def plot_feature_importance(
    model_path="models/tuned_xgboost_model.pkl",
    data_path="data/processed/featured_creditcard.csv"
):
    print("Generating Feature Importance Plot...")

    model = joblib.load(model_path)
    feature_list = joblib.load("models/feature_list.pkl")

    df = pd.read_csv(data_path)
    X = df[feature_list]

    importances = model.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": feature_list,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    os.makedirs("outputs/plots", exist_ok=True)

    plt.figure(figsize=(10, 8))
    plt.barh(
        importance_df["Feature"][:15],
        importance_df["Importance"][:15]
    )
    plt.gca().invert_yaxis()
    plt.title("Top 15 Feature Importances (XGBoost)")
    plt.savefig("outputs/plots/feature_importance.png")
    plt.close()

    print("Feature importance plot saved successfully.")