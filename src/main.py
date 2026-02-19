from eda import run_eda
from preprocessing import preprocess_data
from feature_engineering import engineer_features
from train_models import train_models
from evaluation import evaluate_models
from feature_importance import plot_feature_importance
from shap_explain import generate_shap_explanations
from hyperparameter_tuning import tune_xgboost


def run_pipeline():

    print("\n==============================")
    print("Starting Fraud Detection Pipeline")
    print("==============================\n")

    # -----------------------------------
    # 1. Exploratory Data Analysis
    # -----------------------------------
    print("\n[1/8] Running EDA...")
    run_eda()

    # -----------------------------------
    # 2. Data Preprocessing
    # -----------------------------------
    print("\n[2/8] Preprocessing Data...")
    preprocess_data()

    # -----------------------------------
    # 3. Feature Engineering
    # -----------------------------------
    print("\n[3/8] Engineering Features...")
    engineer_features()

    # -----------------------------------
    # 4. Train Baseline Models
    # -----------------------------------
    print("\n[4/8] Training Logistic & XGBoost...")
    X_test, y_test, y_prob_log, y_prob_xgb = train_models()

    # -----------------------------------
    # 5. Model Evaluation + Threshold Optimization
    # -----------------------------------
    print("\n[5/8] Evaluating Models...")
    evaluate_models(y_test, y_prob_log, y_prob_xgb)

    # -----------------------------------
    # 6. Hyperparameter Tuning (XGBoost)
    # -----------------------------------
    print("\n[6/8] Hyperparameter Tuning...")
    tune_xgboost()

    # -----------------------------------
    # 7. Feature Importance
    # -----------------------------------
    print("\n[7/8] Generating Feature Importance...")
    plot_feature_importance()

    # -----------------------------------
    # 8. SHAP Explainability
    # -----------------------------------
    print("\n[8/8] Generating SHAP Explanations...")
    generate_shap_explanations()

    print("\n==============================")
    print("Pipeline Completed Successfully ✅")
    print("==============================\n")


if __name__ == "__main__":
    run_pipeline()