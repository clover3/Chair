import os

from cpath import output_path
from misc_lib import exist_or_mkdir


def get_genex_run_save_dir():
    genex_run_save_dir = os.path.join(output_path, "genex", "runs")
    exist_or_mkdir(genex_run_save_dir)
    return genex_run_save_dir