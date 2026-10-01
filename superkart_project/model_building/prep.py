
# for data manipulation
import pandas as pd
import sklearn

# for creating a folder
import os

# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split

# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
# api = HfApi(token=os.getenv("HF_TOKEN"))
# DATASET_PATH = "hf://datasets/ChandanKeskar2007/SuperKart-Sales-Prediction/SuperKart.csv"
# superkart_dataset = pd.read_csv(DATASET_PATH)
# print("Dataset loaded successfully.")

api = HfApi(token=os.getenv("HF_TOKEN"))

HF_USER = "ChandanKeskar2007"
DATA_REPO = f"{HF_USER}/SuperKart-Sales-Prediction"

# Load the source dataset directly from Hugging Face
DATASET_PATH = f"hf://datasets/{DATA_REPO}/SuperKart.csv"
superkart_dataset = pd.read_csv(DATASET_PATH)

print("Dataset loaded successfully.")
print("Original shape:", superkart_dataset.shape)

# -------------------------------------------------------
# Data Cleaning
# -------------------------------------------------------

# Standardize an inconsistent category observed in the dataset
superkart_dataset["Product_Sugar_Content"] = (
    superkart_dataset["Product_Sugar_Content"]
    .astype(str)
    .str.strip()
    .replace({"reg": "Regular"})
)

# Unique identifier columns are not used as predictive inputs
columns_to_drop = ["Product_Id", "Store_Id"]
superkart_dataset = superkart_dataset.drop(columns=columns_to_drop)

# Target variable
target = "Product_Store_Sales_Total"

# Numerical features
numeric_features = [
    "Product_Weight",               # Weight of the product.
    "Product_Allocated_Area",       # Ratio of display area allocated to the product.
    "Product_MRP",                  # Maximum retail price of the product.
    "Store_Establishment_Year",     # Year in which the store was established.
]

# Categorical features
categorical_features = [
    "Product_Sugar_Content",        # Sugar content category of the product.
    "Product_Type",                 # Broad product category.
    "Store_Size",                   # Size category of the store.
    "Store_Location_City_Type",     # City tier in which the store is located.
    "Store_Type",                   # Type of store.
]

# Define predictor matrix (X) using selected numeric and categorical features
X = superkart_dataset[numeric_features + categorical_features]

# Define target variable
y = superkart_dataset[target]

print("Cleaned feature shape:", X.shape)
print("Missing values in features:", int(X.isnull().sum().sum()))
print("Missing values in target:", int(y.isnull().sum()))

# -------------------------------------------------------
# Train/Test Split
# -------------------------------------------------------

# Split dataset into train and test
# Split the dataset into training and test sets
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X,y,                # Predictors (X) and target variable (y)
    test_size=0.20,     # 20% of the data is reserved for testing
    random_state=42     # Ensures reproducibility by setting a fixed random seed
)

print("Xtrain shape:", Xtrain.shape)
print("Xtest shape:", Xtest.shape)

# Save locally inside the GitHub Actions runner
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

# Upload split datasets back to Hugging Face
files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path,
        repo_id=DATA_REPO,
        repo_type="dataset",
    )

print("Prepared train/test datasets uploaded successfully.")
