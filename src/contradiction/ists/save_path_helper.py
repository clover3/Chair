import os

from cpath import output_path
from misc_lib import exist_or_mkdir


def get_save_path(save_name):
    dir_path = os.path.join(output_path, "ists", "noali_pred")
    exist_or_mkdir(dir_path)
    save_path = os.path.join(dir_path, save_name + ".txt")
    return save_path


def get_qrel_path(genre, split):
    dir_path = os.path.join(output_path, "ists", "noali_label")
    save_path = os.path.join(dir_path, f"{genre}_{split}.qrel")
    return save_path
