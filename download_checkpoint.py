import os
from huggingface_hub import snapshot_download
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(".env"))
MODEL_DIR = os.getenv("MODEL_DIR")
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
if HUGGINGFACE_TOKEN is None or HUGGINGFACE_TOKEN == "":
    raise ValueError("HUGGINGFACE_TOKEN is not set in .env file.")

def download_realarchvis():
    output_dir = os.path.join(MODEL_DIR, "realarchvis_xlV30.safetensors")
    
    if os.path.exists(output_dir):
        print(f"{output_dir} already exists. Skipping download.")
        return True
    
    try:
        print(f"Downloading sd_controlnet_depth from HuggingFace ...")
        snapshot_download(
            token=HUGGINGFACE_TOKEN,
            repo_id="KaceCM/KandyAIJob",
            repo_type="model",
            allow_patterns=["realarchvis_xlV30.safetensors"],
            local_dir=MODEL_DIR)

        print(f"Download completed and saved to {output_dir}.")
        return True
    except Exception as e:
        print(f"An error occurred while downloading: {e}")
        return False
    
if __name__ == "__main__":
    download_realarchvis()