import streamlit as st
import pickle
import pandas as pd
import numpy as np

st.set_page_config(page_title="SCF Credit Risk Evaluator", page_icon="💳", layout="centered")

# Premium Animated Cyber-Luxury Dark Theme
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(-45deg, #0f172a, #1e1b4b, #3b0764, #111827);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
        color: #ffffff;
    }
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    div[data-testid="stVerticalBlock"] > div {
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(168, 85, 247, 0.3);
        border-radius: 16px;
        padding: 25px;
        margin-bottom: 15px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.4);
        backdrop-filter: blur(5px);
    }
    label, p, h3 { color: #f8fafc !important; font-weight: 600 !important; }
    .stButton>button {
        background-image: linear-gradient(to right, #a855f7 0%, #7e22ce 100%);
        color: #ffffff !important;
        font-weight: bold;
        border-radius: 10px;
        width: 100%;
        padding: 12px;
        border: none;
        box-shadow: 0 4px 15px rgba(168, 85, 247, 0.4);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(168, 85, 247, 0.6);
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center; color:#a855f7;'>💳 Federal Reserve SCF Loan Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#94a3b8; margin-bottom:30px;'>Advanced Automated Institutional Underwriting Simulation</p>", unsafe_allow_html=True)

@st.cache_resource
def load_model_package():
    with open("loan_model.pkl", "rb") as f:
        return pickle.load(f)

try:
    package = load_model_package()
    model = package['model']
    feature_names = package['features']

    st.markdown("### 📊 Financial Risk Indicators")
    
    # Financial Inputs
    income = st.number_input("💵 Annual Family Income ($)", min_value=0, value=75000, step=5000)
    networth = st.number_input("🏦 Estimated Household Net Worth ($)", value=150000, step=10000)
    debt2inc = st.slider("📉 Debt-to-Income Ratio", 0.0, 5.0, 0.35, step=0.05)
    
    st.markdown("### 👤 Applicant Demographics & Credit History")
    
    # Demographic & Credit History Inputs
    age = st.slider("🎂 Age of Head of Household", 18, 95, 40)
    kids = st.number_input("👶 Number of Dependents (Children)", min_value=0, max_value=10, value=0, step=1)
    
    ed_choice = st.selectbox("🎓 Education Level", ["No High School Diploma", "High School Diploma", "Some College / Associate Degree", "College Degree or Higher"])
    ed_map = {"No High School Diploma": 1, "High School Diploma": 2, "Some College / Associate Degree": 3, "College Degree or Higher": 4}
    
    mar_choice = st.selectbox("💍 Marital Status", ["Married", "Unmarried / Single"])
    mar_map = {"Married": 1, "Unmarried / Single": 2}
    
    late_choice = st.selectbox("⚠️ Have you been 60+ days late on payments in the past year?", ["No", "Yes"])
    late_map = {"No": 0, "Yes": 1}

    if st.button("Run Risk Analysis Assessment"):
        # Map values to match model columns in exact order
        input_data = pd.DataFrame([[
            float(income), 
            int(age), 
            float(networth), 
            float(debt2inc), 
            ed_map[ed_choice], 
            mar_map[mar_choice],
            int(kids),
            late_map[late_choice]
        ]], columns=feature_names)

        # 1. Calculate probabilities using built-in matrix extractor
        probabilities = model.predict_proba(input_data)[0]
        
        # In SCF: 0 = Approved, 1 = Denied
        approval_prob = probabilities[0] * 100
        
        # 2. Extract final hard classification label
        prediction = model.predict(input_data)[0]

        st.markdown(f"### 🎯 Underwriting Score Assessment")

        if prediction == 0:
            st.markdown(f"""
                <div style='background-color: rgba(22, 163, 74, 0.25); border-left: 6px solid #22c55e; padding: 20px; border-radius: 12px; margin-top: 10px;'>
                    <h3 style='margin:0; color:#22c55e;'>✅ Credit Application Approved</h3>
                    <p style='margin:5px 0 15px 0; color:#bbf7d0;'>The risk score is well within institutional boundaries. Low default probability detected.</p>
                    <h2 style='margin:0; color:#ffffff;'>👍 Approval Probability: {approval_prob:.1f}%</h2>
                </div>
            """, unsafe_allow_html=True)
            st.progress(int(approval_prob))
        else:
            st.markdown(f"""
                <div style='background-color: rgba(239, 68, 68, 0.25); border-left: 6px solid #ef4444; padding: 20px; border-radius: 12px; margin-top: 10px;'>
                    <h3 style='margin:0; color:#ef4444;'>❌ Credit Application Denied</h3>
                    <p style='margin:5px 0 15px 0; color:#fecaca;'>Elevated delinquency flags or debt-to-income imbalances present an excessive risk profile.</p>
                    <h2 style='margin:0; color:#ffffff;'>👎 Approval Probability: {approval_prob:.1f}%</h2>
                </div>
            """, unsafe_allow_html=True)
            st.progress(int(approval_prob))

except Exception as e:
    st.error(f"System Operational Error: {e}")
