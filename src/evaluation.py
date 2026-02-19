import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    f1_score,
    recall_score,
    precision_recall_curve,
    auc
)

import os


def evaluate_models(y_test, y_prob_log, y_prob_xgb):

    os.makedirs("outputs/plots", exist_ok=True)

    print("\n=== Logistic Regression Report ===")
    print(classification_report(y_test, y_prob_log > 0.5))

    print("\n=== XGBoost Report ===")
    print(classification_report(y_test, y_prob_xgb > 0.5))

    # ROC Curve
    fpr_log, tpr_log, _ = roc_curve(y_test, y_prob_log)
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_prob_xgb)

    plt.figure()
    plt.plot(fpr_log, tpr_log, label="Logistic")
    plt.plot(fpr_xgb, tpr_xgb, label="XGBoost")
    plt.legend()
    plt.title("ROC Curve")
    plt.savefig("outputs/plots/roc_curve.png")
    plt.close()

    # Precision-Recall Curve
    prec_log, rec_log, _ = precision_recall_curve(y_test, y_prob_log)
    prec_xgb, rec_xgb, _ = precision_recall_curve(y_test, y_prob_xgb)

    plt.figure()
    plt.plot(rec_log, prec_log, label="Logistic")
    plt.plot(rec_xgb, prec_xgb, label="XGBoost")
    plt.legend()
    plt.title("Precision-Recall Curve")
    plt.savefig("outputs/plots/pr_curve.png")
    plt.close()

    print("Evaluation plots saved.")


def optimize_threshold(y_test, y_prob, model_name="Model"):
    print(f"\nOptimizing threshold for {model_name}...")

    thresholds = np.arange(0.1, 0.9, 0.05)

    best_threshold = 0.5
    best_f1 = 0

    for t in thresholds:
        preds = (y_prob > t).astype(int)
        f1 = f1_score(y_test, preds)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    print(f"Best Threshold: {best_threshold}")
    print(f"Best F1 Score: {best_f1}")

    recall = recall_score(y_test, (y_prob > best_threshold))
    print(f"Recall at Best Threshold: {recall}")

    return best_threshold