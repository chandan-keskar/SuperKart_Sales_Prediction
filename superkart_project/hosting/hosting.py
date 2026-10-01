from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import os

HF_USER = "ChandanKeskar2007"
SPACE_REPO = f"{HF_USER}/SuperKart-Sales-Prediction"

hf_token = os.getenv("HF_TOKEN")
api = HfApi(token=hf_token)

# Create the Docker Space if it does not already exist
try:
    api.repo_info(repo_id=SPACE_REPO, repo_type="space")
    print(f"Space '{SPACE_REPO}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{SPACE_REPO}' not found. Creating it...")
    create_repo(
        repo_id=SPACE_REPO,
        repo_type="space",
        space_sdk="docker",
        private=False,
        token=hf_token
    )
    print(f"Space '{SPACE_REPO}' created.")

# Upload deployment files to the root of the Hugging Face Space
api.upload_folder(
    folder_path="superkart_project/deployment",
    repo_id=SPACE_REPO,
    repo_type="space",
    path_in_repo="",
)

print("Deployment files uploaded successfully to Hugging Face Space.")
