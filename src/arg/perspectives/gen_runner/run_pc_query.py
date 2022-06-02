import os
from functools import partial

from arg.perspectives.pc_run_path import get_query_file_for_split
from dataset_specific.clue_path import index_name_list
from galagos import query_to_all_clueweb_disk
from list_lib import lmap
from misc_lib import exist_or_mkdir


# query is made from query_gen.py


def num_query_file_for_split(split: str) -> int:
    return {
        "train": 122,
        "dev": 139,
        "test": 0,
    }[split]


def work():
    split = "dev"
    save_parent = '/mnt/nfs/work3/youngwookim/data/perspective/{}_claim_perspective'.format(split)
    exist_or_mkdir(save_parent)
    ranked_list_save_root = os.path.join(save_parent, "q_res_11")
    exist_or_mkdir(ranked_list_save_root)

    get_query_file_fn = partial(get_query_file_for_split, split)
    num_query_file: int = num_query_file_for_split(split)
    query_files = lmap(get_query_file_fn, range(0, num_query_file))
    query_to_all_clueweb_disk.send(query_files,
                                   index_name_list[:1],
                                   "perspective_{}_claim_perspective_11".format(split),
                                   ranked_list_save_root)


if __name__ == "__main__":
    work()
