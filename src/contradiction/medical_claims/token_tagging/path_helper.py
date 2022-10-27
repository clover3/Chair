import os

from cpath import output_path
from misc_lib import exist_or_mkdir


def get_save_path(save_name):
    save_path = os.path.join(output_path, "alamri_annotation1", "ranked_list", save_name + ".txt")
    return save_path


def get_save_path2(run_name, tag_type):
    return get_save_path("{}_{}".format(run_name, tag_type))


def get_binary_save_path(run_name, tag_type):
    save_name = "{}_{}".format(run_name, tag_type)
    dir_save_path = os.path.join(output_path, "alamri_annotation1", "binary_predictions")
    exist_or_mkdir(dir_save_path)
    save_path = os.path.join(dir_save_path, save_name + ".txt")
    return save_path


def get_binary_save_path_w_opt(run_name, tag_type, metric):
    save_name = "{}_{}_{}".format(run_name, tag_type, metric)
    dir_save_path = os.path.join(output_path, "alamri_annotation1", "binary_predictions")
    exist_or_mkdir(dir_save_path)
    save_path = os.path.join(dir_save_path, save_name + ".txt")
    return save_path


def get_sbl_label_json_path():
    save_dir = os.path.join(output_path, "alamri_annotation1", "label", "sel_by_longest.json")
    return save_dir


def get_sbl2_label_json_path():
    save_dir = os.path.join(output_path, "alamri_annotation1", "label", "sel_by_longest2.json")
    return save_dir


def get_sbl_vak_qrel_path():
    return os.path.join(output_path, "alamri_annotation1", "label", "sbl.qrel.val")


def get_sbl_qrel_path(split):
    return os.path.join(output_path, "alamri_annotation1", "label", "sbl.qrel.{}".format(split))


def get_sbl_binary_label_path(tag, split):
    file_path = os.path.join(output_path, "alamri_annotation1", "label", "sbl_binary.{}.{}.jsonl".format(tag, split))
    return file_path
