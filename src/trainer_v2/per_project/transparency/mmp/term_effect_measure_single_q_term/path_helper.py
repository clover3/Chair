import pickle

from cpath import output_path
from misc_lib import path_join


def get_precompute_ranked_list_save_path(corpus_name, key):
    save_dir = path_join(output_path, "msmarco", "passage", "prl")
    save_name = f"{key}_{corpus_name}.pickle"
    save_path = path_join(save_dir, save_name)
    return save_path


def get_gain_save_path(corpus_name, key):
    save_dir = path_join(output_path, "galign", "partial")
    save_name = f"{key}_{corpus_name}.pickle"
    save_path = path_join(save_dir, save_name)
    return save_path


def save_gain(obj, corpus_name, run_name):
    pickle.dump(obj, open(get_gain_save_path(corpus_name, run_name), "wb"))