import os

from cpath import output_path
from misc_lib import exist_or_mkdir
from cpath import output_path
from misc_lib import path_join


def get_evidence_selected0_path(split):
    dir_path = os.path.join(output_path, "align", "evidence_select0")
    exist_or_mkdir(dir_path)
    save_path = os.path.join(dir_path, split)
    return save_path


def get_mmp_es0_path(job_no):
    corpus_path = path_join(output_path, "msmarco", "passage")
    dir_path = path_join(corpus_path, "evidence_select0")
    exist_or_mkdir(dir_path)
    save_path = os.path.join(dir_path, f"{job_no}")
    return save_path
