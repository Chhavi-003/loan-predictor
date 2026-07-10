import streamlit as st
import pickle
import pandas as pd
import numpy as np

# 1. Page Config
st.set_page_config(page_title="SCF Credit Underwriting Engine", page_icon="💳", layout="centered")

# 2. Premium Animated Dark Gold and Charcoal Luxury Gradient Style
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(-45deg, #0f172a, #1e1b4b, #451a03, #111827);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
        color: #ffffff;
    }
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    div[data-testid="stVerticalBlock"] > div:has(div.stNumberInput) {
        background: rgba(15, 23, 42, 0.65);
        border: 1px solid rgba(234, 179, 8, 0.3);
        border-radius: 16px;
        padding: 30px;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5);
        backdrop-filter: blur(8px);
    }
    label, p, h3 {
        color: #f8fafc !important;
        font-weight: 600 !important;
    }
    .stButton>button {
        background-image: linear-gradient(to right, #eab308 0%, #ca8a04 100%);
        color: #0f172a !important;
        font-weight: bold;
        border-radius: 10px;
        width: 100%;
        border: none;
        padding: 12px;
        box-shadow: 0 4px 15px rgba(234, 179, 8, 0.3);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(234, 179, 8, 0.5);
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center; color:#eab308;'>💳 SCF Loan Approval Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#94a3b8; margin-bottom:30px;'>Federal Reserve Survey-backed Underwriting Simulation Tool</p>", unsafe_allow_html=True)

# 3. Load Model
@st.cache_resource
def load_model():
    with open("loan_model.pkl", "rb") as f:
        return pickle.load(f)

try:
    model = load_model()
except Exception as e:
    st.error("Underwriting model brain 'loan_model.pkl' is missing from directory.")

# 4. Input Panel Matching core SCF Metric Columns
st.markdown("### 📊 Enter Applicant Metrics")

income = st.number_input("💵 Total Family Income (INCOME)", min_value=0, value=75000, step=5000)
age = st.slider("🎂 Age of Head of Household (AGE)", 18, 95, 40)
networth = st.number_input("🏦 Household Net Worth (NETWORTH)", value=150000, step=10000)
debt_to_income = st.number_input("📉 Debt-to-Income Ratio (DEBT2INC)", min_value=0.0, max_value=10.0, value=0.35, step=0.05)

# 5. Pipeline Run Execution
if st.button("Evaluate Credit Risk Application"):
    try:
        # Create a complete row of 0s for all 300+ features
        # Assuming your training frame shape matches the columns list provided
        all_columns = ['YY1', 'Y1', 'WGT', 'HHSEX', 'AGE', 'AGECL', 'EDUC', 'EDCL', 'MARRIED', 'KIDS', 'LF', 'LIFECL', 'FAMSTRUCT', 'RACECL', 'RACECL4', 'RACE', 'OCCAT1', 'OCCAT2', 'INDCAT', 'FOODHOME', 'FOODAWAY', 'FOODDELV', 'RENT', 'INCOME', 'WAGEINC', 'BUSSEFARMINC', 'INTDIVINC', 'KGINC', 'SSRETINC', 'TRANSFOTHINC', 'PENACCTWD', 'NORMINC', 'WSAVED', 'SAVED', 'SAVRES1', 'SAVRES2', 'SAVRES3', 'SAVRES4', 'SAVRES5', 'SAVRES6', 'SAVRES7', 'SAVRES8', 'SAVRES9', 'SPENDMOR', 'SPENDLESS', 'EXPENSHILO', 'LATE', 'LATE60', 'HPAYDAY', 'BNKRUPLAST5', 'KNOWL', 'YESFINRISK', 'NOFINRISK', 'CRDAPP', 'TURNDOWN', 'FEARDENIAL', 'TURNFEAR', 'FORECLLAST5', 'EMERGBORR', 'EMERGSAV', 'EMERGPSTP', 'EMERGCUT', 'HBORRFF', 'HBORRCC', 'HBORRALT', 'HBORRFIN', 'HSAVFIN', 'HSAVNFIN', 'HPSTPPAY', 'HPSTPLN', 'HPSTPOTH', 'HCUTFOOD', 'HCUTENT', 'HCUTOTH', 'FINLIT', 'BSHOPNONE', 'BSHOPGRDL', 'BSHOPMODR', 'ISHOPNONE', 'ISHOPGRDL', 'ISHOPMODR', 'BCALL', 'BMAGZNEWS', 'BMAILADTV', 'BINTERNET', 'BFRIENDWORK', 'BFINPRO', 'BSELF', 'BDONT', 'BOTHER', 'ICALL', 'IMAGZNEWS', 'IMAILADTV', 'IINTERNET', 'IFRIENDWORK', 'IFINPRO', 'ISELF', 'IDONT', 'IOTHER', 'BFINPLAN', 'IFINPLAN', 'INTERNET', 'CHECKING', 'HCHECK', 'NOCHK', 'EHCHKG', 'WHYNOCKG', 'DONTWRIT', 'MINBAL', 'DONTLIKE', 'SVCCHG', 'CANTMANG', 'NOMONEY', 'CREDIT', 'DONTWANT', 'OTHER', 'CKLOCATION', 'CKLOWFEEBAL', 'CKMANYSVCS', 'CKRECOMFRND', 'CKPERSONAL', 'CKCONNECTN', 'CKLONGTIME', 'CKSAFETY', 'CKCONVPAYRL', 'CKOTHCHOOSE', 'PREPAID', 'SAVING', 'HSAVING', 'MMDA', 'MMMF', 'MMA', 'HMMA', 'CALL', 'HCALL', 'LIQ', 'HLIQ', 'CDS', 'HCDS', 'STMUTF', 'TFBMUTF', 'GBMUTF', 'OBMUTF', 'COMUTF', 'OMUTF', 'NMMF', 'HNMMF', 'STOCKS', 'HSTOCKS', 'NSTOCKS', 'WILSH', 'NOTXBND', 'MORTBND', 'GOVTBND', 'OBND', 'BOND', 'HBOND', 'IRAKH', 'THRIFT', 'FUTPEN', 'CURRPEN', 'RETQLIQ', 'HRETQLIQ', 'ANYPEN', 'DBPLANCJ', 'DCPLANCJ', 'DBPLANT', 'BPLANCJ', 'SAVBND', 'HSAVBND', 'CASHLI', 'HCASHLI', 'ANNUIT', 'TRUSTS', 'OTHMA', 'HOTHMA', 'OTHFIN', 'HOTHFIN', 'EQUITY', 'HEQUITY', 'DEQ', 'RETEQ', 'EQUITINC', 'HBROK', 'HTRAD', 'NTRAD', 'FIN', 'HFIN', 'VEHIC', 'HVEHIC', 'BUSVEH', 'NBUSVEH', 'OWN', 'NOWN', 'LEASE', 'NLEASE', 'VLEASE', 'NVEHIC', 'NEWCAR1', 'NEWCAR2', 'FARMBUS', 'HOUSES', 'HHOUSES', 'HOUSECL', 'ORESRE', 'HORESRE', 'NNRESRE', 'HNNRESRE', 'BUS', 'ACTBUS', 'NONACTBUS', 'HBUS', 'OTHNFIN', 'HOTHNFIN', 'NFIN', 'HNFIN', 'NHNFIN', 'ASSET', 'HASSET', 'HELOC', 'MRTHEL', 'NH_MORT', 'HOMEEQ', 'HMRTHEL', 'HHELOC', 'HNH_MORT', 'HPRIM_MORT', 'PURCH1', 'REFIN_EVER', 'HEXTRACT_EVER', 'HSEC_MORT', 'PURCH2', 'HMORT2', 'HELOC_YN', 'OTHLOC', 'HOTHLOC', 'MORT1', 'MORT2', 'MORT3', 'RESDBT', 'HRESDBT', 'CCBAL', 'NOCCBAL', 'HCCBAL', 'VEH_INST', 'EDN_INST', 'INSTALL', 'OTH_INST', 'HVEH_INST', 'HEDN_INST', 'HOTH_INST', 'HINSTALL', 'ODEBT', 'HODEBT', 'DEBT', 'HDEBT', 'NETWORTH', 'LEVRATIO', 'DEBT2INC', 'KGHOUSE', 'KGORE', 'KGBUS', 'FARMBUS_KG', 'KGSTMF', 'KGTOTAL', 'PAYMORT1', 'PAYMORT2', 'PAYMORT3', 'PAYMORTO', 'PAYLOC1', 'PAYLOC2', 'PAYLOC3', 'PAYLOCO', 'PAYHI1', 'PAYHI2', 'PAYLC1', 'PAYLC2', 'PAYLCO', 'PAYORE1', 'PAYORE2', 'PAYOREV', 'PAYORE3', 'PAYVEH1', 'PAYVEH2', 'PAYVEH3', 'PAYVEH4', 'PAYVEHM', 'PAYVEO1', 'PAYVEO2', 'PAYVEOM', 'PAYEDU1', 'PAYEDU2', 'PAYEDU3', 'PAYEDU4', 'PAYEDU5', 'PAYEDU6', 'PAYEDU7', 'PAYILN1', 'PAYILN2', 'PAYILN3', 'PAYILN4', 'PAYILN5', 'PAYILN6', 'PAYILN7', 'PAYMARG', 'PAYINS', 'PAYPEN1', 'PAYPEN2', 'PAYPEN3', 'PAYPEN4', 'PAYPEN5', 'PAYPEN6', 'TPAY', 'MORTPAY', 'CONSPAY', 'REVPAY', 'PIRTOTAL', 'PIRMORT', 'PIRCONS', 'PIRREV', 'PIR40', 'PLOAN1', 'PLOAN2', 'PLOAN3', 'PLOAN4', 'PLOAN5', 'PLOAN6', 'PLOAN7', 'PLOAN8', 'LLOAN1', 'LLOAN2', 'LLOAN3', 'LLOAN4', 'LLOAN5', 'LLOAN6', 'LLOAN7', 'LLOAN8', 'LLOAN9', 'LLOAN10', 'LLOAN11', 'LLOAN12', 'NWCAT', 'INCCAT', 'ASSETCAT', 'NINCCAT', 'NINC2CAT', 'NWPCTLECAT', 'INCPCTLECAT', 'NINCPCTLECAT', 'INCQRTCAT', 'NINCQRTCAT']
        
        input_row = pd.DataFrame(0, index=[0], columns=all_columns)
        
        # Inject the active interface selections into the payload row
        input_row['INCOME'] = float(income)
        input_row['AGE'] = int(age)
        input_row['NETWORTH'] = float(networth)
        input_row['DEBT2INC'] = float(debt_to_income)
        
        prediction = model.predict(input_row)[0]
        
        if prediction == 1:
            st.markdown("""
                <div style='background-color: rgba(22, 163, 74, 0.2); border-left: 6px solid #16a34a; padding: 20px; border-radius: 12px; margin-top: 20px;'>
                    <h3 style='margin:0; color:#16a34a;'>✅ Application Approved</h3>
                    <p style='margin:5px 0 0 0; color:#bbf7d0;'>The financial profile clears the underwriting threshold guidelines successfully.</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div style='background-color: rgba(220, 38, 38, 0.2); border-left: 6px solid #dc2424; padding: 20px; border-radius: 12px; margin-top: 20px;'>
                    <h3 style='margin:0; color:#dc2424;'>❌ Application Denied / Rejected</h3>
                    <p style='margin:5px 0 0 0; color:#fecaca;'>Risk profile indicates elevated defaults probability based on historical indicators.</p>
                </div>
            """, unsafe_allow_html=True)
            
    except Exception as e:
        st.error(f"Prediction Pipeline Error. Details: {e}")