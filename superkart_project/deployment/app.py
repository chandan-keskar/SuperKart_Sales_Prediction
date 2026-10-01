import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

HF_USER = "ChandanKeskar2007"
MODEL_REPO = f"{HF_USER}/superkart-sales-model"
MODEL_FILE = "best_superkart_sales_model_v1.joblib"

st.set_page_config(
    page_title="SuperKart Sales Prediction",
    page_icon="🛒",
    layout="centered"
)

@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE
    )
    return joblib.load(model_path)

model = load_model()

st.title("🛒 SuperKart Sales Prediction")
st.write(
    "Enter the product and store details below to estimate the "
    "total sales revenue for the selected product-store combination."
)

# -------------------------------------------------------
# Product Inputs
# -------------------------------------------------------
st.subheader("Product Details")

Product_Weight = st.number_input(
    "Product Weight",
    min_value=4.0,
    max_value=22.0,
    value=12.66,
    step=0.1
)

Product_Sugar_Content = st.selectbox(
    "Product Sugar Content",
    ["Low Sugar", "Regular", "No Sugar"]
)

Product_Allocated_Area = st.number_input(
    "Product Allocated Area Ratio",
    min_value=0.004,
    max_value=0.298,
    value=0.068,
    step=0.001,
    format="%.3f"
)

Product_Type = st.selectbox(
    "Product Type",
    [
        "Baking Goods",
        "Breads",
        "Breakfast",
        "Canned",
        "Dairy",
        "Frozen Foods",
        "Fruits and Vegetables",
        "Hard Drinks",
        "Health and Hygiene",
        "Household",
        "Meat",
        "Others",
        "Seafood",
        "Snack Foods",
        "Soft Drinks",
        "Starchy Foods",
    ]
)

Product_MRP = st.number_input(
    "Product MRP",
    min_value=31.0,
    max_value=266.0,
    value=147.0,
    step=1.0
)

# -------------------------------------------------------
# Store Inputs
# -------------------------------------------------------
st.subheader("Store Details")

Store_Establishment_Year = st.selectbox(
    "Store Establishment Year",
    [1987, 1998, 1999, 2009]
)

Store_Size = st.selectbox(
    "Store Size",
    ["High", "Medium", "Small"]
)

Store_Location_City_Type = st.selectbox(
    "Store Location City Type",
    ["Tier 1", "Tier 2", "Tier 3"]
)

Store_Type = st.selectbox(
    "Store Type",
    [
        "Departmental Store",
        "Food Mart",
        "Supermarket Type1",
        "Supermarket Type2",
    ]
)

# -------------------------------------------------------
# Construct input dataframe
# -------------------------------------------------------
input_data = pd.DataFrame([{
    "Product_Weight": Product_Weight,
    "Product_Allocated_Area": Product_Allocated_Area,
    "Product_MRP": Product_MRP,
    "Store_Establishment_Year": Store_Establishment_Year,
    "Product_Sugar_Content": Product_Sugar_Content,
    "Product_Type": Product_Type,
    "Store_Size": Store_Size,
    "Store_Location_City_Type": Store_Location_City_Type,
    "Store_Type": Store_Type,
}])

if st.button("Predict Sales"):
    prediction = float(model.predict(input_data)[0])

    st.success("Sales prediction generated successfully.")
    st.metric(
        label="Predicted Product Store Sales Total",
        value=f"{prediction:,.2f}"
    )
