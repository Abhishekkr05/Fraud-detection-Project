# ================================
# FRAUD DETECTION STREAMLIT APP
# ================================

# Import required libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------
# Page Configuration
# --------------------------------
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    layout="wide",
    page_icon="💳"
)

# --------------------------------
# Custom CSS Styling (Makes UI Attractive)
# --------------------------------
st.markdown("""
<style>
.main {
    background-color: #f4f6f9;
}
h1 {
    color: #1f77b4;
}
.stButton>button {
    background-color: #1f77b4;
    color: white;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------
# Load Saved Model & Preprocessor
# --------------------------------
# (These were saved earlier using joblib)
model = joblib.load("fraud_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

# --------------------------------
# App Title
# --------------------------------
st.title("💳 Fraud Detection ML Dashboard")
st.markdown("### Real-time Transaction Fraud Prediction System")

# --------------------------------
# Sidebar Input Section
# --------------------------------
st.sidebar.header("🔍 Enter Transaction Details")

account_age_days = st.sidebar.number_input("Account Age (days)", min_value=0)
total_transactions_user = st.sidebar.number_input("Total Transactions", min_value=0)
avg_amount_user = st.sidebar.number_input("Average User Amount", min_value=0.0)
amount = st.sidebar.number_input("Transaction Amount", min_value=0.0)
shipping_distance_km = st.sidebar.number_input("Shipping Distance (km)", min_value=0.0)

country = st.sidebar.text_input("Country")
bin_country = st.sidebar.text_input("BIN Country")
channel = st.sidebar.selectbox("Channel", ["web", "mobile", "pos"])
merchant_category = st.sidebar.text_input("Merchant Category")
promo_used = st.sidebar.selectbox("Promo Used", ["yes", "no"])
avs_match = st.sidebar.selectbox("AVS Match", ["Y", "N"])
cvv_result = st.sidebar.selectbox("CVV Result", ["M", "N"])
three_ds_flag = st.sidebar.selectbox("3DS Flag", ["Y", "N"])

# --------------------------------
# Create Input DataFrame
# --------------------------------
input_data = pd.DataFrame({
    "account_age_days": [account_age_days],
    "total_transactions_user": [total_transactions_user],
    "avg_amount_user": [avg_amount_user],
    "amount": [amount],
    "shipping_distance_km": [shipping_distance_km],
    "country": [country],
    "bin_country": [bin_country],
    "channel": [channel],
    "merchant_category": [merchant_category],
    "promo_used": [promo_used],
    "avs_match": [avs_match],
    "cvv_result": [cvv_result],
    "three_ds_flag": [three_ds_flag]
})

# --------------------------------
# Prediction Button
# --------------------------------
if st.sidebar.button("🚀 Predict Fraud"):

    # Apply preprocessing (scaling + encoding)
    input_processed = preprocessor.transform(input_data)

    # Get prediction
    prediction = model.predict(input_processed)[0]
    probability = model.predict_proba(input_processed)[0][1]

    st.subheader("📊 Prediction Result")

    # Display result
    if prediction == 1:
        st.error("⚠️ Fraudulent Transaction Detected!")
    else:
        st.success("✅ Legitimate Transaction")

    # --------------------------------
    # Probability Gauge
    # --------------------------------
    st.write("### Fraud Probability")
    st.progress(float(probability))

    st.write(f"Fraud Probability: **{probability:.2%}**")

# ==========================================
# MODEL PERFORMANCE SECTION
# ==========================================

st.markdown("---")
st.header("📈 Model Performance Insights")

# Sample static metrics (replace with your real values)
accuracy = 0.94
precision = 0.91
recall = 0.88
f1 = 0.89

col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", accuracy)
col2.metric("Precision", precision)
col3.metric("Recall", recall)
col4.metric("F1 Score", f1)

# --------------------------------
# Confusion Matrix Visualization
# --------------------------------
st.subheader("Confusion Matrix")

cm = np.array([[850, 50],
               [40, 60]])  # Example values

fig1, ax1 = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax1)
ax1.set_xlabel("Predicted")
ax1.set_ylabel("Actual")

st.pyplot(fig1)

# --------------------------------
# Feature Importance (Random Forest)
# --------------------------------
st.subheader("Feature Importance")

# Extract feature importance if model is RandomForest
if hasattr(model, "feature_importances_"):
    importances = model.feature_importances_
    feature_names = preprocessor.get_feature_names_out()

    feature_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).head(10)

    fig2, ax2 = plt.subplots()
    sns.barplot(x="Importance", y="Feature", data=feature_df, ax=ax2)
    st.pyplot(fig2)

# --------------------------------
# Footer
# --------------------------------
st.markdown("---")
st.markdown("Developed by Abhishek Kumar | CSE | Machine Learning Project 🚀")
