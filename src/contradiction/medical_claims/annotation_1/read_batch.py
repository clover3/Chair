import os

from cpath import output_path
from misc_lib import get_dir_files


def load_file_list():
    dir_path = os.path.join(output_path, "alamri_annotation1", "batch_results")
    files = get_dir_files(dir_path)
    return files