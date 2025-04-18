# app.py
import streamlit as st
import numpy as np
import pickle

# Load model and scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('kmeans_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("🧠 Customer Segmentation Predictor")

st.markdown("Enter customer behavior data to get the segment:")

payment = st.number_input("Average Payment Value (R$)", min_value=0.0, step=1.0)
orders = st.number_input("Number of Orders", min_value=0)
review = st.slider("Average Review Score", 0.0, 5.0, 4.0)
freight = st.number_input("Average Freight Value (R$)", min_value=0.0, step=1.0)

if st.button("Predict Segment"):
    user_input = np.array([[payment, orders, review, freight]])
    user_scaled = scaler.transform(user_input)
    cluster = model.predict(user_scaled)[0]
    
    st.success(f"This customer belongs to Cluster {cluster}")
    
    # Optional: add interpretation
    if cluster == 0:
        st.info("🟢 Likely a loyal customer with high satisfaction.")
    elif cluster == 1:
        st.info("🟡 New or budget-conscious customer.")
    elif cluster == 2:
        st.info("🔴 High freight cost, low reviews — risky customer.")
    else:
        st.info("🔵 Cluster behavior under analysis.")

