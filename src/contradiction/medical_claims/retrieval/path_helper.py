from cpath import output_path
from misc_lib import path_join


def get_retrieval_save_path(save_name):
    save_path = path_join(output_path, "alamri_annotation1", "retrieval", save_name + ".txt")
    return save_path


def get_retrieval_qrel_path(split):
    save_path = path_join(output_path, "alamri_annotation1", "retrieval_qrel", split + ".qrel")
    return save_path

