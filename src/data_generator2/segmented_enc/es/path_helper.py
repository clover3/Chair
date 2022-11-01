import os

from cpath import output_path
from misc_lib import exist_or_mkdir


def get_evidence_selected0_path(split):
    dir_path = os.path.join(output_path, "align", "evidence_select0")
    exist_or_mkdir(dir_path)
    save_path = os.path.join(dir_path, split)
    return save_path
