from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os

# repo_id = "<-----Hugging Face User ID ----->/SuperKart-Sales-Prediction"
HF_USER = "ChandanKeskar2007"
repo_id = f"{HF_USER}/SuperKart-Sales-Prediction"
repo_type = "dataset"

# Initialize API client
# hf_token = os.getenv("HF_TOKEN")
# api = HfApi(token=HF_TOKEN)
api = HfApi(token=os.getenv("HF_TOKEN"))

# Step 1: Check if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Space '{repo_id}' created.")

api.upload_folder(
    folder_path="superkart_project/data",
    repo_id=repo_id,
    repo_type=repo_type,
)


print("Dataset uploaded successfully.")
