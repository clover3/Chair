
from cpath import output_path
from misc_lib import path_join, exist_or_mkdir

TASK_NAME = "pte_scientsbank"


def get_score_save_path(save_name):
    exist_or_mkdir(path_join(output_path, TASK_NAME))
    exist_or_mkdir(path_join(output_path, TASK_NAME,  "prediction_scores",))
    save_path = path_join(output_path, TASK_NAME, "prediction_scores", save_name + ".txt")
    return save_path


def get_threshold_save_path(save_name):
    exist_or_mkdir(path_join(output_path, TASK_NAME,  "threshold",))
    save_path = path_join(output_path, TASK_NAME, "threshold", save_name + ".txt")
    return save_path



