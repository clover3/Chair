import os

from cpath import output_path
from misc_lib import exist_or_mkdir


no_tune_method_list = ["exact_match", "davinci", "gpt-3.5-turbo"]
mismatch_only_method_list = ["exact_match", "word2vec_em", "coattention"]


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
    if run_name in no_tune_method_list:
        save_name = "{}_{}".format(run_name, tag_type)
    else:
        save_name = "{}_{}_{}".format(run_name, tag_type, metric)
    dir_save_path = os.path.join(output_path, "alamri_annotation1", "binary_predictions")
    exist_or_mkdir(dir_save_path)
    save_path = os.path.join(dir_save_path, save_name + ".txt")
    return save_path


def get_flat_binary_save_path(run_name, tag_type, metric):
    save_name = "{}_{}_{}".format(run_name, tag_type, metric)
    dir_save_path = os.path.join(output_path, "alamri_annotation1", "flat_binary_predictions")
    exist_or_mkdir(dir_save_path)
    save_path = os.path.join(dir_save_path, save_name + ".txt")
    return save_path


def get_debug_qid_save_path(run_name, tag_type, metric):
    save_name = "{}_{}_{}".format(run_name, tag_type, metric)
    dir_save_path = os.path.join(output_path, "alamri_annotation1", "debug_qid")
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
