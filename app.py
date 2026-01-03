import streamlit as st
import pandas as pd
import joblib

model = joblib.load('models/model.pkl')
model_columns = joblib.load('models/columns.pkl')

st.set_page_config(page_title="Churn Predictor", layout="centered")
st.title("📊 Customer Churn Risk Analyzer")

col1, col2 = st.columns(2)

with col1:
    tenure = st.slider("Tenure (Months)", 1, 72, 1)
    monthly_charges = st.slider("Monthly Charges ($)", 18, 120, 110)

with col2:
    contract = st.selectbox("Contract Type", ["Month-to-Month", "One Year", "Two Year"])
    internet = st.selectbox("Internet Type", ["Fiber Optic", "DSL", "Cable", "None"])

total_charges = tenure * monthly_charges

input_dict = {
    'Tenure in Months': tenure,
    'Monthly Charge': monthly_charges,
    'Total Charges': total_charges
}

input_df = pd.DataFrame([input_dict])

for col in model_columns:
    if col not in input_df.columns:
        input_df[col] = 0

if f"Contract_{contract}" in model_columns:
    input_df[f"Contract_{contract}"] = 1
if f"Internet Type_{internet}" in model_columns:
    input_df[f"Internet Type_{internet}"] = 1

input_df = input_df[model_columns]

if st.button("Analyze Risk Level"):
    prediction = model.predict_proba(input_df)[0][1]
    
    st.markdown(f"### Probability of Leaving: **{prediction:.2%}**")
    
    if prediction > 0.60:
        st.error("🚨 **HIGH RISK**: Urgent intervention recommended. High likelihood of cancellation.")
    elif prediction > 0.30:
        st.warning("⚠️ **MEDIUM RISK**: Customer is at risk. Consider offering a loyalty discount or contract upgrade.")
    else:
        st.success("✅ **LOW RISK**: Customer appears stable and loyal.")
        
    st.info("Note: This prediction is based on financial and contract patterns, excluding satisfaction surveys.")