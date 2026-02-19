import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import os


def generate_shap_explanations(
    model_path="models/tuned_xgboost_model.pkl",
    data_path="data/processed/featured_creditcard.csv"
):
    print("Generating SHAP explanations...")

    model = joblib.load(model_path)
    feature_list = joblib.load("models/feature_list.pkl")

    df = pd.read_csv(data_path)
    X = df[feature_list]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X.sample(1000, random_state=42))

    os.makedirs("outputs/plots", exist_ok=True)

    shap.summary_plot(shap_values, X.sample(1000, random_state=42), show=False)
    plt.savefig("outputs/plots/shap_summary.png")
    plt.close()

    print("SHAP summary plot saved.")