import os
from utils.tqdm_utils import download_with_tqdm, extract_zip_file_with_tqdm


def download_extract(url, extract_dir, zip_file_save_name):
    # Create the directory if it doesn't exist
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir)
    file_path = os.path.join(extract_dir, zip_file_save_name)
    print(f"Downloading file from {url}")
    download_with_tqdm(url, file_path)
    # Extract the zip file with progress bar
    print(f"Extracting files to {extract_dir}")
    extract_zip_file_with_tqdm(file_path, extract_dir)
    # Delete the zip file
    os.remove(file_path)
    print("File downloaded and extracted successfully.")
