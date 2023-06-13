import os.path

from cpath import data_path
from misc_lib import path_join


def get_semeval_2013_task7_path():
    semeval_2013_task7_dir = path_join(data_path, "semeval-2013-task7")
    return semeval_2013_task7_dir


def get_semeval_2013_task7_dataset():
    semeval_2013_task7_dir = get_semeval_2013_task7_path()
    if not os.path.exists(semeval_2013_task7_dir):
        print("Data directory does not exists. Downloading from web")
        download_semeval_5way_data()
    return semeval_2013_task7_dir


def get_pte_dataset(split_name):
    split_dir = path_join(get_semeval_2013_task7_dataset(), "sciEntsBank", split_name)
    pte_dir = path_join(split_dir, "Extra")
    return pte_dir


def download_semeval_5way_data():
    from dataset_specific.scientsbank.download_data import download_extract
    # URL of the file to download
    url = "https://github.com/myrosia/semeval-2013-task7/raw/main/semeval-5way.zip"
    zip_file_save_name = "semeval-5way.zip"
    semeval_2013_task7_dir = get_semeval_2013_task7_path()
    extract_dir = semeval_2013_task7_dir

    download_extract(url, extract_dir, zip_file_save_name)