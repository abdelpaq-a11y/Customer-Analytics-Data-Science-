import streamlit as st
import pandas as pd
import joblib

# ==============================
# Page Configuration
# ==============================
st.set_page_config(
    page_title="Electronics Purchase Prediction",
    page_icon="🛒",
    layout="centered"
)

# ==============================
# Load Trained Model
# ==============================
import joblib
import streamlit as st

@st.cache_resource
def load_model():
    return joblib.load("electronics_rf_pipeline.joblib")

model = load_model()


# ==============================
# Title & Description
# ==============================
st.title("🛒 Electronics Purchase Prediction System")

st.markdown("""
This application predicts whether a customer purchase will be **Electronics** or **Other Product**
using a **Random Forest Classifier** trained on historical customer data.
""")

# ==============================
# Sidebar – Model Info
# ==============================
with st.sidebar:
    st.header("ℹ️ Model Information")
    st.markdown("""
    **Target Variable:**  
    `is_electronics`  

    **Input Features:**  
    - Age  
    - Purchase Amount  
    - Rating  
    - Purchase Month  
    - Gender  

    **Preprocessing:**  
    - StandardScaler (Numerical)  
    - OneHotEncoder (Gender)  

    **Model:**  
    - RandomForestClassifier
    """)

# ==============================
# User Inputs
# ==============================
st.subheader("🧾 Enter Customer Data")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age",
        min_value=10,
        max_value=100,
        value=30
    )

    purchase_amount = st.number_input(
        "Purchase Amount",
        min_value=0.0,
        max_value=10000.0,
        value=500.0
    )

    rating = st.slider(
        "Product Rating",
        min_value=1.0,
        max_value=5.0,
        value=4.0
    )

with col2:
    month = st.selectbox(
        "Purchase Month",
        options=list(range(1, 13))
    )

    gender = st.selectbox(
        "Gender",
        options=["male", "female"]
    )

# ==============================
# Prediction
# ==============================
if st.button("🔍 Predict Purchase Type", use_container_width=True):

    input_data = pd.DataFrame([{
        "Age": age,
        "PurchaseAmount": purchase_amount,
        "Rating": rating,
        "Month": month,
        "Gender": gender
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("📊 Prediction Result")

    if prediction == 1:
        st.error(
            f"🖥️ **Electronics Purchase**\n\n"
            f"Probability: **{probability:.2%}**"
        )
    else:
        st.success(
            f"🛍️ **Other Product**\n\n"
            f"Probability: **{1 - probability:.2%}**"
        )

    st.progress(int(probability * 100))

    st.info(
        "Prediction is based on customer behavior patterns learned from historical data."
    )
if st.button("🔍 Predict Purchase Type", use_container_width=True):

    input_data = pd.DataFrame([{
        "Age": age,
        "PurchaseAmount": purchase_amount,
        "Rating": rating,
        "Month": month,
        "Gender": gender
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("📊 Prediction Result")

    if prediction == 1:
        st.error(
            f"🖥️ **Electronics Purchase**\n\n"
            f"Probability: **{probability:.2%}**"
        )
    else:
        st.success(
            f"🛍️ **Other Product**\n\n"
            f"Probability: **{1 - probability:.2%}**"
        )

    st.progress(int(probability * 100))

    st.info(
        "Prediction is based on customer behavior patterns learned from historical data."
    )

# To run it use this command : python -m streamlit run "C:\Users\Omar AbdElpaq\OneDrive\Desktop\Final Projects\DS\Final Version\app.py"