import os

from cpath import output_path
from misc_lib import exist_or_mkdir


def get_ists_save_path(genre, split, run_name) -> str:
    dir_path = os.path.join(output_path, "ists")
    exist_or_mkdir(dir_path)
    return os.path.join(dir_path, f"{genre}.{split}.{run_name}.txt")


def get_ists_2d_save_path(genre, split, run_name) -> str:
    dir_path = os.path.join(output_path, "ists", "2d")
    exist_or_mkdir(dir_path)
    return os.path.join(dir_path, f"{genre}.{split}.{run_name}.txt")