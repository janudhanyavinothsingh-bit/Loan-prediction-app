import streamlit as st
import pandas as pd
import joblib

# Load the saved model
model = joblib.load("loan_prediction_model.pkl")

st.title("🏦 Loan Approval Prediction App")

# Collect user input
no_of_dependents = st.sidebar.number_input("Number of Dependents", min_value=0, step=1)
education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.sidebar.selectbox("Self Employed", ["Yes", "No"])
income_annum = st.sidebar.number_input("Annual Income", min_value=0, step=1000)
loan_amount = st.sidebar.number_input("Loan Amount", min_value=0, step=1000)
loan_term = st.sidebar.number_input("Loan Term (months)", min_value=1, step=1)
cibil_score = st.sidebar.number_input("CIBIL Score", min_value=300, max_value=900, step=1)
residential_assets_value = st.sidebar.number_input("Residential Assets Value", min_value=0, step=1000)
commercial_assets_value = st.sidebar.number_input("Commercial Assets Value", min_value=0, step=1000)
luxury_assets_value = st.sidebar.number_input("Luxury Assets Value", min_value=0, step=1000)
bank_asset_value = st.sidebar.number_input("Bank Asset Value", min_value=0, step=1000)

# Convert categorical inputs
education_val = 1 if education == "Graduate" else 0
self_employed_val = 1 if self_employed == "Yes" else 0

# Prepare input data with EXACT feature names
applicant_data = {
    'loan_id': 9999,  # dummy ID
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

# Force column order
applicant_df = pd.DataFrame([applicant_data])[model.feature_names_in_]

# Add a threshold slider in the sidebar
threshold = st.sidebar.slider("Approval Threshold (%)", 0, 100, 50, 5)

# Initialize session state for logging
if "results" not in st.session_state:
    st.session_state["results"] = []
# Prediction button
if st.button("Predict Loan Status"):
    proba = model.predict_proba(applicant_df)[0]
    st.write(f"Approval Probability: {proba[1]*100:.2f}%")

   # Combined decision logic
if cibil_score < 500 or income_annum < 5000:
    st.error("❌ Loan Not Approved (Rule-based)")
    decision = "Not Approved (Rule-based)"
elif proba[1]*100 > threshold:
    st.success("✅ Loan Approved")
    decision = "Approved"
else:
    st.error("❌ Loan Not Approved")
    decision = "Not Approved"

    # Log the result into session state
    st.session_state["results"].append({
        "Dependents": no_of_dependents,
        "Education": education,
        "Self Employed": self_employed,
        "Income": income_annum,
        "Loan Amount": loan_amount,
        "CIBIL": cibil_score,
        "Probability": f"{proba[1]*100:.2f}%",
        "Decision": decision
    })

# Show results table
if "results" in st.session_state and st.session_state["results"]:
    st.write("### Prediction History")
    st.dataframe(pd.DataFrame(st.session_state["results"]))

    # Clear history button
    if st.button("Clear History"):
        st.session_state["results"] = []
        st.success("History cleared!")

# Add explanation button separately
if st.button("Explain Prediction"):
    importances = model.feature_importances_
    features = model.feature_names_in_
    importance_df = pd.DataFrame({
        "Feature": features,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)
    st.bar_chart(importance_df.set_index("Feature"))
