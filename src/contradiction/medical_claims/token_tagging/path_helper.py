import os

from cpath import output_path


def get_save_path(save_name):
    save_path = os.path.join(output_path, "alamri_annotation1", "ranked_list", save_name + ".txt")
    return save_path


def get_save_path2(run_name, tag_type):
    return get_save_path("{}_{}".format(run_name, tag_type))