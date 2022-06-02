import os
from functools import partial
from typing import Iterable, Counter

from arg.perspectives.clueweb_db import preload_tf, load_tf
from arg.perspectives.pc_run_path import train_query_indices
from cache import load_from_pickle
from dataset_specific.clue_path import index_name_list
from galagos.parse import merge_ranked_list_list, load_galago_ranked_list
from list_lib import l_to_map, lmap


class CollectionInterface:
    def __init__(self, ranked_list_save_root, do_lowercase=True):
        print("CollectionInterface __init__")
        self.available_disk_list = index_name_list[:1]
        load_all_ranked_list_fn = partial(load_all_ranked_list, ranked_list_save_root)
        self.ranked_list = l_to_map(load_all_ranked_list_fn, self.available_disk_list)
        self.do_lowercase = do_lowercase
        self.collection_tf = load_from_pickle("collection_tf")
        self.collection_ctf = sum(self.collection_tf.values())
        print("CollectionInterface __init__ Done")

    def get_ranked_documents_tf(self, claim_id, perspective_id, allow_not_found=False) -> Iterable[Counter]:
        query_id = "{}_{}".format(claim_id, perspective_id)
        ranked_list = self.get_ranked_list(query_id)
        doc_ids = lmap(lambda x: x[0], ranked_list)
        preload_tf(doc_ids)

        def do_load_tf(doc_id):
            try:
                counter = load_tf(doc_id)
                if self.do_lowercase:
                    counter = {key.lower(): value for key, value in counter.items()}
            except KeyError:
                if allow_not_found:
                    counter = None
                else:
                    raise
            return counter

        return lmap(do_load_tf, doc_ids)

    def get_ranked_list(self, query_id):
        ranked_list_list = []
        last_error = None
        for disk_name in self.available_disk_list:
            try:
                ranked_list = self.ranked_list[disk_name][query_id]
                ranked_list_list.append(ranked_list)
            except KeyError as e:
                print(e)
                last_error = e
                pass
        if not ranked_list_list:
            raise last_error

        return merge_ranked_list_list(ranked_list_list)

    def tf_collection(self, term):
        return self.collection_tf[term], self.collection_ctf


def load_all_ranked_list(ranked_list_save_root, disk_name):
    d = {}
    for idx in train_query_indices:
        file_name = "{}_{}.txt".format(disk_name, idx)
        file_path = os.path.join(ranked_list_save_root, file_name)
        d.update(load_galago_ranked_list(file_path))

    return d