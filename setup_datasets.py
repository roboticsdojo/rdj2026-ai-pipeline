import os
import shutil
import zipfile
import subprocess
import glob

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "dataset", "raw")
ARCHIVE_ZIP = os.path.join(os.path.expanduser("~"), "Downloads", "archive.zip")
TRASHNET_URL = "https://github.com/garythung/trashnet.git"

def main():
    print("Setting up datasets...")
    os.makedirs(os.path.join(RAW_DIR, "watermelon"), exist_ok=True)
    os.makedirs(os.path.join(RAW_DIR, "aluminium_foil"), exist_ok=True)
    os.makedirs(os.path.join(RAW_DIR, "plastics"), exist_ok=True)

    # 1. Handle Kaggle Archive
    if os.path.exists(ARCHIVE_ZIP):
        print(f"Extracting {ARCHIVE_ZIP}...")
        tmp_kaggle = "/tmp/kaggle_data"
        if os.path.exists(tmp_kaggle):
            shutil.rmtree(tmp_kaggle)
        os.makedirs(tmp_kaggle, exist_ok=True)
        with zipfile.ZipFile(ARCHIVE_ZIP, 'r') as zip_ref:
            zip_ref.extractall(tmp_kaggle)
        
        # Kaggle classes
        # biological -> watermelon
        # (Skipped: We only want watermelon images here, not general biological waste)
        biological_dir = os.path.join(tmp_kaggle, "garbage_classification", "biological")
        # if os.path.exists(biological_dir):
        #     for f in glob.glob(os.path.join(biological_dir, "*.jpg")):
        #         shutil.copy(f, os.path.join(RAW_DIR, "watermelon", os.path.basename(f)))
        print("Note: Skipped importing Kaggle's general 'biological' waste. Please add your watermelon images to dataset/raw/watermelon/")
        
        # metal -> aluminium_foil
        # (Skipped: We only want aluminium foil images here, not general metal waste)
        metal_dir = os.path.join(tmp_kaggle, "garbage_classification", "metal")
        # if os.path.exists(metal_dir):
        #     for f in glob.glob(os.path.join(metal_dir, "*.jpg")):
        #         shutil.copy(f, os.path.join(RAW_DIR, "aluminium_foil", os.path.basename(f)))
        print("Note: Skipped importing Kaggle's general 'metal' waste. Please add your aluminium foil images to dataset/raw/aluminium_foil/")

        # plastic -> plastics
        plastic_dir = os.path.join(tmp_kaggle, "garbage_classification", "plastic")
        if os.path.exists(plastic_dir):
            for f in glob.glob(os.path.join(plastic_dir, "*.jpg")):
                shutil.copy(f, os.path.join(RAW_DIR, "plastics", os.path.basename(f)))
                
        shutil.rmtree(tmp_kaggle)
        print("Kaggle data merged successfully.")
    else:
        print(f"Warning: Could not find Kaggle zip at {ARCHIVE_ZIP}.")
        print("Please place the downloaded archive.zip there or adapt this script.")

    # 2. Handle TrashNet
    tmp_trashnet = "/tmp/trashnet_repo"
    if not os.path.exists(tmp_trashnet):
        print(f"Cloning {TRASHNET_URL}...")
        subprocess.run(["git", "clone", TRASHNET_URL, tmp_trashnet], check=True)
    
    trashnet_zip = os.path.join(tmp_trashnet, "data", "dataset-resized.zip")
    if os.path.exists(trashnet_zip):
        print(f"Extracting TrashNet data from {trashnet_zip}...")
        with zipfile.ZipFile(trashnet_zip, 'r') as zip_ref:
            zip_ref.extractall(tmp_trashnet)
        
        # (Skipped: We only want aluminium foil here)
        trashnet_metal = os.path.join(tmp_trashnet, "dataset-resized", "metal")
        # if os.path.exists(trashnet_metal):
        #     for f in glob.glob(os.path.join(trashnet_metal, "*.jpg")):
        #         shutil.copy(f, os.path.join(RAW_DIR, "aluminium_foil", "trashnet_" + os.path.basename(f)))
                
        trashnet_plastic = os.path.join(tmp_trashnet, "dataset-resized", "plastic")
        if os.path.exists(trashnet_plastic):
            for f in glob.glob(os.path.join(trashnet_plastic, "*.jpg")):
                shutil.copy(f, os.path.join(RAW_DIR, "plastics", "trashnet_" + os.path.basename(f)))
        
        print("TrashNet data merged successfully.")
    else:
        print("Could not find dataset-resized.zip in TrashNet clone.")

    print(f"\nData setup complete! Images are now in {RAW_DIR}.")
    print("You can now run 'python3 prepare_dataset.py' to build the final dataset.")

if __name__ == "__main__":
    main()
