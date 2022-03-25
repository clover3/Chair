import os

from epath import job_man_dir


def get_segmented_data_path(dataset_name, split, job_id):
    file_path = os.path.join(job_man_dir, f"{dataset_name}_{split}_tokenize", str(job_id))
    return file_path