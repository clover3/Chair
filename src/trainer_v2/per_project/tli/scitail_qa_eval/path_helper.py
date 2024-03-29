from cpath import output_path
from misc_lib import path_join


def get_score_save_path(save_name):
    save_path = path_join(output_path, "scitail_qa", "prediction_scores", save_name + ".txt")
    return save_path


def get_ranked_list_save_path(save_name):
    save_path = path_join(output_path, "scitail_qa", "ranked_list", save_name + ".txt")
    return save_path


def get_qrel_path(save_name):
    save_path = path_join(output_path, "scitail_qa", "labels", save_name + ".txt")
    return save_path

