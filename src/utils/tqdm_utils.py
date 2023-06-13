import zipfile

import requests
from tqdm import tqdm


def download_with_tqdm(url, file_path):
    # Download the zip file
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    with open(file_path, "wb") as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    return response


def extract_zip_file_with_tqdm(file_path, extract_dir):
    zip_file = zipfile.ZipFile(file_path, "r")
    total_files = len(zip_file.infolist())
    progress_bar = tqdm(total=total_files, unit='file')
    for file in zip_file.infolist():
        zip_file.extract(file, extract_dir)
        progress_bar.update(1)
    progress_bar.close()