import os

from cpath import output_path


def get_save_path(save_name):
    save_path = os.path.join(output_path, "ists", "noali_pred", save_name + ".txt")
    return save_path
