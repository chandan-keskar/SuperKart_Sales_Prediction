
# for data manipulation
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# for model serialization
import joblib

# for creating a folder
import os

# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
# import mlflow

HF_USER = "ChandanKeskar2007"
DATA_REPO = f"{HF_USER}/SuperKart-Sales-Prediction"
MODEL_REPO = f"{HF_USER}/superkart-sales-model"

# -------------------------------------------------------
# Load train/test data from Hugging Face
# -------------------------------------------------------
Xtrain_path = f"hf://datasets/{DATA_REPO}/Xtrain.csv"
Xtest_path  = f"hf://datasets/{DATA_REPO}/Xtest.csv"
ytrain_path = f"hf://datasets/{DATA_REPO}/ytrain.csv"
ytest_path  = f"hf://datasets/{DATA_REPO}/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path).squeeze("columns")
ytest = pd.read_csv(ytest_path).squeeze("columns")

print("Training data loaded:", Xtrain.shape)
print("Testing data loaded:", Xtest.shape)

numeric_features = [
    "Product_Weight",
    "Product_Allocated_Area",
    "Product_MRP",
    "Store_Establishment_Year",
]

categorical_features = [
    "Product_Sugar_Content",
    "Product_Type",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
]

# -------------------------------------------------------
# Preprocessing
# -------------------------------------------------------

# Define the preprocessing steps
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features)
)

# -------------------------------------------------------
# XGBoost Regression Model
# -------------------------------------------------------
xgb_model = xgb.XGBRegressor(
    objective="reg:squarederror",
    random_state=42,
    n_jobs=2
)

model_pipeline = make_pipeline(preprocessor, xgb_model)

# A practical grid for CI/CD execution
param_grid = {
    "xgbregressor__n_estimators": [100, 200],
    "xgbregressor__max_depth": [3, 5],
    "xgbregressor__learning_rate": [0.05, 0.10],
    "xgbregressor__subsample": [0.8, 1.0],
    "xgbregressor__colsample_bytree": [0.8, 1.0],
}

# -------------------------------------------------------
# Hyperparameter Tuning
# -------------------------------------------------------
grid_search = GridSearchCV(
    estimator=model_pipeline,
    param_grid=param_grid,
    scoring="neg_root_mean_squared_error",
    cv=3,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(Xtrain, ytrain)

print("\\nBest Parameters:")
print(grid_search.best_params_)

best_model = grid_search.best_estimator_

best_cv_rmse = -grid_search.best_score_
print(f"Best Cross-Validation RMSE: {best_cv_rmse:.4f}")

best_model = grid_search.best_estimator_

# -------------------------------------------------------
# Evaluation
# -------------------------------------------------------
train_pred = best_model.predict(Xtrain)
test_pred = best_model.predict(Xtest)

train_rmse = np.sqrt(mean_squared_error(ytrain, train_pred))
test_rmse = np.sqrt(mean_squared_error(ytest, test_pred))

train_mae = mean_absolute_error(ytrain, train_pred)
test_mae = mean_absolute_error(ytest, test_pred)

train_r2 = r2_score(ytrain, train_pred)
test_r2 = r2_score(ytest, test_pred)

print("\\nModel Performance")
print("-----------------")
print(f"Train RMSE: {train_rmse:.4f}")
print(f"Test RMSE : {test_rmse:.4f}")
print(f"Train MAE : {train_mae:.4f}")
print(f"Test MAE  : {test_mae:.4f}")
print(f"Train R2  : {train_r2:.4f}")
print(f"Test R2   : {test_r2:.4f}")

# -------------------------------------------------------
# Save best complete pipeline
# -------------------------------------------------------
model_path = "best_superkart_sales_model_v1.joblib"
joblib.dump(best_model, model_path)
print(f"\\nModel saved locally as: {model_path}")

# -------------------------------------------------------
# Register model on Hugging Face Model Hub
# -------------------------------------------------------

# hf_token = os.getenv("HF_TOKEN")
# api = HfApi(token=hf_token)
api = HfApi()

try:
    api.repo_info(repo_id=MODEL_REPO, repo_type="model")
    print(f"Model repository '{MODEL_REPO}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Model repository '{MODEL_REPO}' not found. Creating it...")
    create_repo(repo_id=MODEL_REPO, repo_type="model", private=False)
    print(f"Model repository '{MODEL_REPO}' created.")

api.upload_file(
    path_or_fileobj=model_path,
    path_in_repo=model_path,
    repo_id=MODEL_REPO,
    repo_type="model",
)

print("Best model registered successfully on Hugging Face.")
