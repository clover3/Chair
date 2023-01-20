import os
from cpath import output_path, data_path
from misc_lib import path_join


def get_save_path(save_name):
    save_path = os.path.join(output_path, "mnli_ex", "ranked_list", save_name + ".txt")
    return save_path


def get_save_path_ex(split, run_name, tag_type):
    save_name = "{}_{}_{}".format(split, run_name, tag_type)
    save_path = get_save_path(save_name)
    return save_path


def get_binary_save_path_w_opt(run_name, tag_type, metric):
    save_name = "{}_{}_{}".format(run_name, tag_type, metric)
    dir_save_path = os.path.join(output_path, "mnli_ex", "binary_predictions")
    save_path = path_join(dir_save_path, save_name + ".txt")
    return save_path


def get_mnli_ex_trec_style_label_path(label, split):
    save_path = os.path.join(
        data_path, "nli", "mnli_ex",
        "trec_style", "{}_{}.txt".format(label, split))
    return save_path