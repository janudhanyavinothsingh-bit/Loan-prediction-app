import streamlit as st
import pandas as pd
import joblib

# Load the saved model
model = joblib.load("loan_prediction_model.pkl")

st.title("🏦 Loan Approval Prediction App")

# Collect user input
no_of_dependents = st.number_input("Number of Dependents", min_value=0, step=1)
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.selectbox("Self Employed", ["Yes", "No"])
income_annum = st.number_input("Annual Income", min_value=0, step=1000)
loan_amount = st.number_input("Loan Amount", min_value=0, step=1000)
loan_term = st.number_input("Loan Term (months)", min_value=1, step=1)
cibil_score = st.number_input("CIBIL Score", min_value=300, max_value=900, step=1)
residential_assets_value = st.number_input("Residential Assets Value", min_value=0, step=1000)
commercial_assets_value = st.number_input("Commercial Assets Value", min_value=0, step=1000)
luxury_assets_value = st.number_input("Luxury Assets Value", min_value=0, step=1000)
bank_asset_value = st.number_input("Bank Asset Value", min_value=0, step=1000)

# Convert categorical inputs
education_val = 1 if education == "Graduate" else 0
self_employed_val = 1 if self_employed == "Yes" else 0

# Prepare input data
applicant_data = {
    'no_of_dependents': no_of_dependents,
    'education': education_val,
    'self_employed': self_employed_val,
    'income_annum': income_annum,
    'loan_amount': loan_amount,
    'loan_term': loan_term,
    'cibil_score': cibil_score,
    'residential_assets_value': residential_assets_value,
    'commercial_assets_value': commercial_assets_value,
    'luxury_assets_value': luxury_assets_value,
    'bank_asset_value': bank_asset_value
}

applicant_df = pd.DataFrame([applicant_data])

# Prediction button
if st.button("Predict Loan Status"):
    prediction = model.predict(applicant_df)[0]
    if prediction == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Not Approved")