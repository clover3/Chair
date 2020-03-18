import os

from list_lib import lmap

ranked_list_save_root = '/mnt/nfs/work3/youngwookim/data/perspective/train_claim_perspective/all_query_results'
query_dir = "/mnt/nfs/work3/youngwookim/code/Chair/output/perspective_train_claim_perspective_query"


train_query_indices = range(0, 122)


def get_query_file(i):
    return os.path.join(query_dir, "{}.json".format(i))


def get_all_query_file_name():
    query_files = lmap(get_query_file, range(0, 122))

##