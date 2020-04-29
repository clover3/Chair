import os
import pickle

from cache import save_to_pickle
from data_generator.job_runner import sydney_working_dir
from list_lib import lmap
from misc_lib import merge_dict_list


def combine_pc_train_info():
    st = 0
    ed = 606

    def load_file(i):
        pickle_path = os.path.join(sydney_working_dir, "pc_rel_tfrecord_info", "{}".format(i))
        return pickle.load(open(pickle_path, "rb"))

    d_list = lmap(load_file, range(st, ed))
    combined_dict = merge_dict_list(d_list)
    save_to_pickle(combined_dict, "pc_rel_info_all")


def combine_pc_dev_info():
    st = 0
    ed = 696

    def load_file(i):
        pickle_path = os.path.join(sydney_working_dir, "pc_rel_tfrecord_dev_info", "{}".format(i))
        return pickle.load(open(pickle_path, "rb"))

    d_list = lmap(load_file, range(st, ed))
    combined_dict = merge_dict_list(d_list)
    save_to_pickle(combined_dict, "pc_rel_dev_info_all")


if __name__ == "__main__":
    combine_pc_dev_info()
