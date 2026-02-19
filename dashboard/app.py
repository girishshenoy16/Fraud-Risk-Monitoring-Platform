import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix

# ----------------------------------------------------
# PAGE CONFIG + DARK FINTECH STYLING
# ----------------------------------------------------
st.set_page_config(layout="wide", page_title="Fraud Monitoring Platform")

st.markdown("""
    <style>
    body {
        background-color: #0e1117;
        color: white;
    }
    .stMetric {
        background-color: #1c1f26;
        padding: 15px;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("💳 Enterprise Fraud Risk Monitoring Platform")

# ----------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------
OPTIMAL_THRESHOLD = 0.30

plt.rcParams["figure.figsize"] = (8, 4)
plt.rcParams["figure.dpi"] = 100


# ----------------------------------------------------
# LOAD MODELS & DATA
# ----------------------------------------------------
@st.cache_resource
def load_models():
    log_model = joblib.load("models/logistic_model.pkl")
    xgb_model = joblib.load("models/tuned_xgboost_model.pkl")
    return log_model, xgb_model

@st.cache_data
def load_data():
    return pd.read_csv("data/processed/featured_creditcard.csv")

log_model, xgb_model = load_models()
df = load_data()

# Model input features (exclude original display columns)
drop_cols = ["Class", "Time_original", "Amount_original"]
model_features = df.drop(columns=[col for col in drop_cols if col in df.columns])

X = model_features

y_true = df["Class"]

# ----------------------------------------------------
# PREDICTIONS
# ----------------------------------------------------
df["Logistic_Risk_Score"] = np.round(log_model.predict_proba(X)[:, 1], 4)
df["Risk Score"] = np.round(xgb_model.predict_proba(X)[:, 1], 4)

df["Fraud Prediction"] = (df["Risk Score"] > OPTIMAL_THRESHOLD).astype(int)

# ----------------------------------------------------
# ALERT SEVERITY CATEGORIES
# ----------------------------------------------------
def risk_category(score):
    if score >= 0.75:
        return "High Risk"
    elif score >= 0.40:
        return "Medium Risk"
    else:
        return "Low Risk"

df["Risk Category"] = df["Risk Score"].apply(risk_category)

# ----------------------------------------------------
# TABS
# ----------------------------------------------------
(
    tab1, tab2, tab3, tab4, tab5,
    tab6, tab7, tab8, tab9, tab10) = (
    st.tabs(
        [
            "📊 Overview",
            "📈 Model Performance",
            "🔎 Feature Insights",
            "🧾 Transaction Explorer",
            "🧠 Explainability",
            "📉 Fraud Trend",
            "💰 Financial Exposure",
            "🥧 Risk Distribution",
            "📊 Executive Summary",
            "📝 Audit Log"
        ]
    )
)


# ====================================================
# TAB 1 — OVERVIEW
# ====================================================
with tab1:
    st.subheader("📊 Risk Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Transactions", f"{len(df):,}")
    col2.metric("Actual Fraud Cases", f"{y_true.sum():,}")
    col3.metric("Detected Fraud Cases", f"{df['Fraud Prediction'].sum():,}")
    col4.metric("Model Threshold", f"{OPTIMAL_THRESHOLD:.2f}")

    st.markdown("### Risk Distribution")

    risk_counts = df["Risk Category"].value_counts()

    risk_df = pd.DataFrame({
        "Risk Category": risk_counts.index,
        "Count": risk_counts.values
    })

    risk_df["Count"] = risk_df["Count"].apply(lambda x: f"{x:,}")

    st.write(risk_df)

    st.write(df.columns)

# ====================================================
# TAB 2 — MODEL PERFORMANCE
# ====================================================
with tab2:
    st.subheader("ROC Curve Comparison")

    fpr_log, tpr_log, _ = roc_curve(y_true, df["Logistic_Risk_Score"])
    fpr_xgb, tpr_xgb, _ = roc_curve(y_true, df["Risk Score"])

    auc_log = round(auc(fpr_log, tpr_log), 2)
    auc_xgb = round(auc(fpr_xgb, tpr_xgb), 2)


    # ROC Curve
    fig1, ax = plt.subplots(figsize=(8, 4))

    plt.title("ROC Curve")

    plt.plot(fpr_log, tpr_log, label=f"Logistic AUC = {auc_log}")
    plt.plot(fpr_xgb, tpr_xgb, label=f"XGBoost AUC = {auc_xgb}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.tight_layout()
    plt.legend()

    st.pyplot(fig1)


    # Precision-Recall Curve
    st.subheader("Precision-Recall Curve")

    prec_log, rec_log, _ = precision_recall_curve(y_true, df["Logistic_Risk_Score"])
    prec_xgb, rec_xgb, _ = precision_recall_curve(y_true, df["Risk Score"])

    fig2, ax = plt.subplots(figsize=(8, 4))

    plt.title("Precision-Recall Curve")

    plt.plot(rec_log, prec_log, label="Logistic")
    plt.plot(rec_xgb, prec_xgb, label="XGBoost")
    plt.tight_layout()
    plt.legend()

    st.pyplot(fig2)


    # Confusion Matrix
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_true, df["Fraud Prediction"])

    fig3, ax = plt.subplots(figsize=(5, 4))

    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.tight_layout()

    st.pyplot(fig3)


# ====================================================
# TAB 3 — FEATURE INSIGHTS
# ====================================================
with tab3:
    st.subheader("Top Feature Drivers (XGBoost)")

    importances = xgb_model.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    # Feature Importance
    fig4, ax = plt.subplots(figsize=(8, 5))

    plt.title("Top 15 Important Features")
    plt.barh(
        importance_df["Feature"][:15],
        importance_df["Importance"][:15]
    )

    plt.gca().invert_yaxis()
    plt.tight_layout()

    st.pyplot(fig4)


# ====================================================
# TAB 4 — TRANSACTION EXPLORER
# ====================================================
with tab4:
    st.subheader("Transaction Risk Explorer")

    filter_option = st.selectbox(
        "Filter by Risk Category",
        ["All", "High Risk", "Medium Risk", "Low Risk"]
    )

    if filter_option != "All":
        filtered_df = df[df["Risk Category"] == filter_option]
    else:
        filtered_df = df

    display_df = filtered_df.copy()

    # Limit rows first
    display_df = display_df.head(50)

    st.markdown(
        """
        ### Risk Legend
        🔴 High Risk  
        🟠 Medium Risk  
        🟢 Low Risk  
        """
    )

    # -------------------------------
    # Show Only Business Columns
    # -------------------------------
    columns_to_show = []

    if "Time_original" in display_df.columns:
        columns_to_show.append("Time_original")

    if "Amount_original" in display_df.columns:
        columns_to_show.append("Amount_original")

    columns_to_show += ["Risk Score", "Risk Category", "Fraud Prediction"]

    display_df = display_df[columns_to_show]

    display_df.rename(columns={
        "Time_original": "Transaction Time",
        "Amount_original": "Transaction Amount"
    }, inplace=True)

    # -------------------------------
    # Highlight Function
    # -------------------------------
    def highlight_risk(row):
        if row["Risk Category"] == "High Risk":
            return ["background-color: #ff4b4b"] * len(row)
        elif row["Risk Category"] == "Medium Risk":
            return ["background-color: #ffa600"] * len(row)
        elif row["Risk Category"] == "Low Risk":
            return ["background-color: #28a745"] * len(row)
        else:
            return [""] * len(row)

    # Apply styling FIRST
    styled_df = display_df.style.apply(highlight_risk, axis=1)

    # Then format numeric columns (2 decimals)
    numeric_cols = display_df.select_dtypes(include=[np.number]).columns
    styled_df = styled_df.format({col: "{:.2f}" for col in numeric_cols})

    st.dataframe(styled_df, use_container_width=True)


# ====================================================
# TAB 5 — SHAP EXPLAINABILITY
# ====================================================
with tab5:

    st.subheader("Transaction-Level Risk Explanation")

    index = st.number_input(
        "Select Transaction Index",
        min_value=0,
        max_value=len(X)-1,
        value=0
    )

    selected_score = round(df.iloc[index]["Risk Score"], 2)
    selected_category = df.iloc[index]["Risk Category"]
    prediction = df.iloc[index]["Fraud Prediction"]

    # ----------------------------------------
    # RISK SUMMARY PANEL
    # ----------------------------------------
    st.markdown("### Risk Assessment Summary")

    col1, col2, col3 = st.columns(3)

    col1.metric("Risk Score", f"{selected_score:.2f}")
    col2.metric("Risk Category", selected_category)
    col3.metric("Fraud Decision",
                "Fraudulent" if prediction == 1 else "Legitimate")

    # ----------------------------------------
    # ACTION RECOMMENDATION
    # ----------------------------------------
    st.markdown("### Recommended Action")

    if selected_category == "High Risk":
        st.error("""
        🔴 HIGH RISK DETECTED  
        - Immediately block transaction  
        - Trigger fraud investigation workflow  
        - Notify customer for verification  
        """)
    elif selected_category == "Medium Risk":
        st.warning("""
        🟠 MEDIUM RISK DETECTED  
        - Flag for manual review  
        - Send verification OTP  
        - Monitor account activity  
        """)
    else:
        st.success("""
        🟢 LOW RISK  
        - Approve transaction  
        - Continue passive monitoring  
        """)

    # ----------------------------------------
    # SHAP EXPLANATION
    # ----------------------------------------
    st.markdown("### Model Explanation (SHAP)")

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X.iloc[[index]])

    plt.figure(figsize=(8, 4))

    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[0],
            base_values=explainer.expected_value,
            data=X.iloc[index],
            feature_names=X.columns
        ),
        show=False
    )

    st.pyplot(plt.gcf(), use_container_width=False)

# ====================================================
# TAB 6 — FRAUD TREND ANALYSIS
# ====================================================
with tab6:
    st.subheader("Fraud Trend Over Time")

    if "Time_original" in df.columns:
        df["Transaction Hour"] = (df["Time_original"] // 3600).astype(int)
        trend = df.groupby("Transaction Hour")["Fraud Prediction"].sum()

        # Fraud Trend Tab
        fig, ax = plt.subplots(figsize=(8, 4))

        plt.title("Fraud Detection Trend by Hour")
        plt.plot(trend.index, trend.values)
        plt.xlabel("Hour")
        plt.ylabel("Detected Fraud Count")
        plt.tight_layout()

        st.pyplot(fig)
    else:
        st.info("Original time data not available.")


# ====================================================
# TAB 7 — FINANCIAL EXPOSURE
# ====================================================
with tab7:
    st.subheader("Estimated Financial Exposure")

    if "Amount_original" in df.columns:
        total_exposure = df[df["Fraud Prediction"] == 1]["Amount_original"].sum()

        st.metric(
            "Estimated Fraud Exposure",
            f"${total_exposure:,.2f}"
        )

        st.write("This represents the potential blocked financial loss.")
    else:
        st.info("Amount data not available.")


# ====================================================
# TAB 8 — RISK DISTRIBUTION PIE CHART
# ====================================================
with tab8:
    st.subheader("Risk Category Distribution")

    risk_counts = df["Risk Category"].value_counts()

    risk_df = pd.DataFrame({
        "Risk Category": risk_counts.index,
        "Count": risk_counts.values
    })

    risk_df["Percentage"] = (
            risk_df["Count"] / risk_df["Count"].sum() * 100
    ).round(2)

    # Risk Distribution Bar Chart
    fig, ax = plt.subplots(figsize=(8, 4))

    colors = {
        "High Risk": "#ff4b4b",
        "Medium Risk": "#ffa600",
        "Low Risk": "#28a745"
    }

    bar_colors = [colors.get(cat, "#888888") for cat in risk_df["Risk Category"]]

    ax.barh(
        risk_df["Risk Category"],
        risk_df["Percentage"],
        color=bar_colors
    )

    ax.set_xlabel("Percentage (%)")
    ax.set_title("Risk Severity Breakdown")

    for i, v in enumerate(risk_df["Percentage"]):
        ax.text(v + 0.1, i, f"{v:.2f}%", va="center")

    plt.tight_layout()
    st.pyplot(fig)


# ====================================================
# TAB 9 — EXECUTIVE SUMMARY
# ====================================================
with tab9:
    st.subheader("Executive Risk Summary")

    total_txn = len(df)
    fraud_detected = df["Fraud Prediction"].sum()
    fraud_rate = (fraud_detected / total_txn) * 100

    st.markdown(
        f"""
        ### 📌 Key Insights
    
        - Total Transactions Processed: **{total_txn:,}**
        - Fraud Cases Detected: **{fraud_detected:,}**
        - Fraud Detection Rate: **{fraud_rate:.2f}%**
        - Model Threshold Used: **{OPTIMAL_THRESHOLD:.2f}**
    
        ### Strategic Observation
    
        - Majority of fraud risk is concentrated in High Risk category.
        - XGBoost model shows strong separation power via ROC and PR curves.
        - Operational risk exposure is actively mitigated through threshold-based blocking.
        """
    )


# ====================================================
# TAB 10 — AUDIT LOG SIMULATION
# ====================================================
with tab10:
    st.subheader("Fraud Detection Audit Log")

    audit_df = df[df["Fraud Prediction"] == 1].copy()

    columns_to_show = []

    if "Time_original" in audit_df.columns:
        columns_to_show.append("Time_original")

    if "Amount_original" in audit_df.columns:
        columns_to_show.append("Amount_original")

    columns_to_show += ["Risk Score", "Risk Category"]

    audit_df = audit_df[columns_to_show]

    audit_df.rename(columns={
        "Time_original": "Transaction Time",
        "Amount_original": "Transaction Amount"
    }, inplace=True)

    numeric_cols = audit_df.select_dtypes(include=[np.number]).columns
    audit_df[numeric_cols] = audit_df[numeric_cols].round(2)

    st.dataframe(audit_df.head(100), use_container_width=True)

    st.download_button(
        "Download Fraud Audit Log",
        audit_df.to_csv(index=False),
        "fraud_audit_log.csv",
        "text/csv"
    )