import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
import joblib
import os


def tune_xgboost(
    input_path="data/processed/featured_creditcard.csv"
):
    print("Starting Hyperparameter Tuning...")

    df = pd.read_csv(input_path)

    drop_cols = ["Class", "Time_original", "Amount_original"]
    X = df.drop(columns=[col for col in drop_cols if col in df.columns])
    y = df["Class"]

    feature_list = X.columns.tolist()
    joblib.dump(feature_list, "models/feature_list.pkl")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    scale_weight = len(y_train[y_train==0]) / len(y_train[y_train==1])

    model = XGBClassifier(
        scale_pos_weight=scale_weight,
        eval_metric="logloss"
    )

    param_grid = {
        "max_depth": [3, 5],
        "learning_rate": [0.01, 0.1],
        "n_estimators": [100, 200]
    }

    grid = GridSearchCV(
        model,
        param_grid,
        scoring="roc_auc",
        cv=3,
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    print("Best Parameters:", grid.best_params_)
    print("Best ROC-AUC:", grid.best_score_)

    os.makedirs("models", exist_ok=True)
    joblib.dump(grid.best_estimator_, "models/tuned_xgboost_model.pkl")

    print("Tuned model saved.")