import path
import os
from misc_lib import *

gsutil_path = "/mnt/scratch/youngwookim/anaconda3/envs/27/bin/gsutil"


def gsutil_cp(src_path, save_dir):
    cmd = "{} cp {} {}".format(gsutil_path, src_path, save_dir)
    os.system(cmd)


def download_model(dir_path, model_step, save_name):
    model_name = "model.ckpt-{}".format(model_step)
    model_path = dir_path + "/" + model_name

    save_dir = os.path.join(path.model_path, "runs", save_name)
    exist_or_mkdir(save_dir)

    src_path = model_path + "*"
    gsutil_cp(src_path, save_dir)

    src_path = dir_path + "/checkpoint"
    gsutil_cp(src_path, save_dir)

    return save_name, model_name