import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
import plotly.express as px

# Import your src modules
from 03_src import 01_data_preprocessing as dp
from 03_src import 02_model_evaluation as me
from 03_src import 03_model_training as mt

# ---------------- Page Config ----------------
st.set_page_config(page_title="Customer Churn Prediction App - Pro+", layout="wide")
st.title("📈 Customer Churn Prediction App - Pro+ Edition")

# ---------------- File Upload ----------------
uploaded_file = st.file_uploader("Upload your telecom CSV file", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.subheader("Data Preview")
    st.dataframe(data.head())

    # ---------------- Data Cleaning ----------------
    data = dp.preprocess_data(data)  # call your preprocessing module
    X, y = dp.feature_engineering(data)  # call feature engineering
    X_res, y_res = dp.handle_imbalance(X, y)  # handle imbalance

    # ---------------- Train/Test Split ----------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
    )

    # ---------------- Model ----------------
    model = mt.train_model(X_train, y_train)  # call training module

    # ---------------- Model Metrics ----------------
    st.subheader("🤖 Model Accuracy & Classification Report")
    y_pred = model.predict(X_test)
    accuracy = model.score(X_test, y_test)
    st.write(f"**Test Accuracy:** {accuracy*100:.2f}%")
    st.text(classification_report(y_test, y_pred))

    # Stratified K-Fold CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_res, y_res, cv=cv, scoring='accuracy')
    st.write(f"**Stratified CV Accuracy:** {np.mean(cv_scores)*100:.2f}% ± {np.std(cv_scores)*100:.2f}%")

    # ---------------- Feature Importance ----------------
    st.subheader("📊 Top 10 Feature Importance (Interactive)")
    importance = model.feature_importances_
    feat_df = pd.DataFrame({"Feature": X.columns, "Importance": importance}).sort_values(by="Importance", ascending=False).head(10)
    fig = px.bar(feat_df, x="Importance", y="Feature", orientation='h', text="Importance")
    st.plotly_chart(fig)

    # ---------------- Sidebar Inputs ----------------
    st.sidebar.header("Advanced Prediction Inputs")
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    internet = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.sidebar.number_input("Monthly Charges", 0.0, 200.0, 50.0)
    risk_thresh = st.sidebar.slider("Risk Threshold (%)", 0, 100, 50)

    # ---------------- Advanced Prediction ----------------
    st.subheader("🔮 Predict Churn (Advanced)")
    if st.button("Predict Churn"):
        sample = dp.create_sample(X.columns, gender, contract, internet, tenure, monthly_charges)
        prob_churn = model.predict_proba(sample)[0][1]*100

        # Color-coded risk
        if prob_churn < risk_thresh*0.5:
            st.success(f"✅ Low Risk: Customer likely to STAY ({prob_churn:.2f}%)")
        elif prob_churn < risk_thresh:
            st.warning(f"⚠️ Medium Risk: Customer may CHURN ({prob_churn:.2f}%)")
        else:
            st.error(f"❌ High Risk: Customer likely to CHURN ({prob_churn:.2f}%)")

        # Customer summary table
        summary = pd.DataFrame({
            "Input": ["Gender","Contract","Internet Service","Tenure","Monthly Charges"],
            "Value": [gender, contract, internet, tenure, monthly_charges],
            "Predicted Churn Probability (%)": [prob_churn]*5
        })
        st.table(summary)

    # ---------------- Batch Prediction ----------------
    st.subheader("📂 Batch Prediction (Upload CSV)")
    batch_file = st.file_uploader("Upload CSV for batch prediction", type=["csv"], key="batch")
    if batch_file is not None:
        batch_data = pd.read_csv(batch_file)
        batch_data = dp.preprocess_batch(batch_data, X.columns)
        batch_preds, batch_probs = mt.batch_predict(model, batch_data, X.columns)
        batch_data["Churn_Prediction"] = np.where(batch_preds==1,"CHURN","STAY")
        batch_data["Churn_Probability"] = batch_probs
        st.dataframe(batch_data)
        st.download_button(
            "📥 Download Predictions CSV",
            data=batch_data.to_csv(index=False),
            file_name="batch_predictions.csv",
            mime="text/csv"
        )
