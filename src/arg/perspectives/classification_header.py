import os

from cpath import data_path
from misc_lib import exist_or_mkdir


def get_file_path(split):
    dir_path = os.path.join(data_path, "perspective_classification")
    exist_or_mkdir(dir_path)
    return os.path.join(dir_path, split + ".csv")