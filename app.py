import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ============================================================
# LOAD MODEL AND TRAINING COLUMNS
# ============================================================
model         = joblib.load("models/best_model.pkl")
train_columns = joblib.load("models/train_columns.pkl")

# Load training data for dropdown options
train_df = pd.read_csv("data/processed/train.csv")
train_df = train_df.drop(columns=["Product_Store_Sales_Total"])

cat_cols = ["Product_Sugar_Content", "Product_Type", "Store_Id",
            "Store_Size", "Store_Location_City_Type", "Store_Type"]

# ============================================================
# APP TITLE
# ============================================================
st.title("🛒 SuperKart Sales Forecasting App")
st.markdown("### Predict Product Store Sales using Machine Learning")
st.markdown("---")

# ============================================================
# GET INPUTS FROM USER
# ============================================================
st.sidebar.header("Enter Product & Store Details")

st.sidebar.subheader("Product Details")
Product_Weight = st.sidebar.number_input(
    "Product Weight", min_value=0.0, max_value=50.0, value=10.0)

Product_Sugar_Content = st.sidebar.selectbox(
    "Product Sugar Content",
    train_df["Product_Sugar_Content"].unique().tolist())

Product_Allocated_Area = st.sidebar.number_input(
    "Product Allocated Area", min_value=0.0, max_value=1.0, value=0.1)

Product_Type = st.sidebar.selectbox(
    "Product Type",
    train_df["Product_Type"].unique().tolist())

Product_MRP = st.sidebar.number_input(
    "Product MRP", min_value=0.0, max_value=500.0, value=100.0)

st.sidebar.subheader("Store Details")
Store_Id = st.sidebar.selectbox(
    "Store ID",
    train_df["Store_Id"].unique().tolist())

Store_Establishment_Year = st.sidebar.number_input(
    "Store Establishment Year",
    min_value=1980, max_value=2025, value=2000)

Store_Size = st.sidebar.selectbox(
    "Store Size",
    train_df["Store_Size"].unique().tolist())

Store_Location_City_Type = st.sidebar.selectbox(
    "Store Location City Type",
    train_df["Store_Location_City_Type"].unique().tolist())

Store_Type = st.sidebar.selectbox(
    "Store Type",
    train_df["Store_Type"].unique().tolist())

# ============================================================
# SAVE INPUTS INTO DATAFRAME
# ============================================================
input_data = pd.DataFrame({
    "Product_Weight"           : [Product_Weight],
    "Product_Sugar_Content"    : [Product_Sugar_Content],
    "Product_Allocated_Area"   : [Product_Allocated_Area],
    "Product_Type"             : [Product_Type],
    "Product_MRP"              : [Product_MRP],
    "Store_Id"                 : [Store_Id],
    "Store_Establishment_Year" : [Store_Establishment_Year],
    "Store_Size"               : [Store_Size],
    "Store_Location_City_Type" : [Store_Location_City_Type],
    "Store_Type"               : [Store_Type]
})

st.subheader("Input Data")
st.dataframe(input_data)

# ============================================================
# ENCODE AND ALIGN INPUT DATA
# ============================================================
input_encoded = pd.get_dummies(input_data, columns=cat_cols)

# Align with exact model columns
input_encoded = input_encoded.reindex(
    columns=train_columns,
    fill_value=0
)

# ============================================================
# PREDICT
# ============================================================
st.markdown("---")
if st.button("🔮 Predict Sales"):
    prediction = model.predict(input_encoded)
    st.success(f"### Predicted Sales: ${prediction[0]:,.2f}")
    st.balloons()

st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | SuperKart MLOps Project")
