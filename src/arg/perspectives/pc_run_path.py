import os

from cpath import output_path
from list_lib import lmap

subproject_hub = '/mnt/nfs/work3/youngwookim/data/perspective/train_claim_perspective/'

ranked_list_save_root = os.path.join(subproject_hub, 'all_query_results')
ranked_list_dir_bm25_0 = os.path.join(subproject_hub, 'bm25_k0_q_res')

query_dir_format = os.path.join(output_path, "perspective_{}_claim_perspective_query_k0")


train_query_indices = range(0, 122)


def get_train_query_file(i):
    return get_query_file(query_dir_format.format("train"), i)


def get_query_file(query_dir, i):
    return os.path.join(query_dir, "{}.json".format(i))


def get_query_file_for_split(split, i):
    return get_query_file(query_dir_format.format(split), i)



def get_all_query_file_name():
    query_files = lmap(get_train_query_file, range(0, 122))

##