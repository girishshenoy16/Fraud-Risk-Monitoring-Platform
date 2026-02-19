import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import joblib
import os


def train_models(
    input_path="data/processed/featured_creditcard.csv"
):
    print("Training Models...")

    df = pd.read_csv(input_path)

    drop_cols = ["Class", "Time_original", "Amount_original"]
    X = df.drop(columns=[col for col in drop_cols if col in df.columns])
    y = df["Class"]

    # Save feature list (CRITICAL)
    os.makedirs("models", exist_ok=True)
    feature_list = X.columns.tolist()
    joblib.dump(feature_list, "models/feature_list.pkl")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    # Logistic
    log_model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000
    )
    log_model.fit(X_train, y_train)
    y_prob_log = log_model.predict_proba(X_test)[:, 1]
    print("Logistic ROC-AUC:", round(roc_auc_score(y_test, y_prob_log), 4))
    joblib.dump(log_model, "models/logistic_model.pkl")

    # XGBoost
    scale_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])

    xgb_model = XGBClassifier(
        scale_pos_weight=scale_weight,
        eval_metric="logloss"
    )

    xgb_model.fit(X_train, y_train)
    y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
    print("XGBoost ROC-AUC:", round(roc_auc_score(y_test, y_prob_xgb), 4))

    joblib.dump(xgb_model, "models/xgboost_model.pkl")

    return X_test, y_test, y_prob_log, y_prob_xgb


if __name__ == "__main__":
    train_models()