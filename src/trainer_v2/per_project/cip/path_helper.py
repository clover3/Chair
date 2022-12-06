import os

from cpath import output_path
from misc_lib import exist_or_mkdir


def get_nlits_segmentation_trial_save_path(split):
    save_dir = os.path.join(output_path, "nlits")
    exist_or_mkdir(save_dir)
    return os.path.join(save_dir, "segmentations_{}.jsonl".format(split))


def get_nli_baseline_pred_save_path(split):
    save_dir = os.path.join(output_path, "nlits")
    exist_or_mkdir(save_dir)
    return os.path.join(save_dir, "baseline_{}.jsonl".format(split))


def get_nlits_segmentation_trial_subjob_save_dir():
    save_dir = os.path.join(output_path, "nlits", "subjob")
    exist_or_mkdir(save_dir)
    return save_dir


def get_cip_dataset_path(name, split):
    save_dir = os.path.join(output_path, "nlits", name)
    exist_or_mkdir(save_dir)
    return os.path.join(save_dir, split)

