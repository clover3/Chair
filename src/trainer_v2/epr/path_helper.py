import os

from cpath import output_path
from epath import job_man_dir


def get_segmented_data_path(dataset_name, split, job_id):
    file_path = os.path.join(job_man_dir, f"{dataset_name}_{split}_tokenize", str(job_id))
    return file_path


def get_output_epr_root():
    return os.path.join(output_path, "epr")


def get_alignment_path(dataset_name, split, job_id):
    file_path = os.path.join(get_output_epr_root(), f"{dataset_name}_{split}_sbert_align", str(job_id))
    return file_path