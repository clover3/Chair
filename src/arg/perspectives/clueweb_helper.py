import os

from datastore.interface import load, save, load_multiple, preload_man
from datastore.table_names import TokenizedCluewebDoc, CluewebDocTF
from galagos.basic import load_galago_ranked_list


# Ranked list from Clueweb with respect to each of claims
class ClaimRankedList:
    def __init__(self):
        dir_path = "/mnt/nfs/work3/youngwookim/data/perspective/dev_perspective/ranked_list"
        disk_name = "ClueWeb12-Disk1_00.idx"
        ranked_list_file_path = os.path.join(dir_path, disk_name)
        self.ranked_list = load_galago_ranked_list(ranked_list_file_path)

    def get(self, cid):
        return self.ranked_list[cid]


def load_doc(doc_id):
    return load(TokenizedCluewebDoc, doc_id)


def load_tf(doc_id):
    return load(CluewebDocTF, doc_id)


def preload_tf(doc_ids):
    preload_man.preload(CluewebDocTF, doc_ids)


def preload_docs(doc_ids):
    preload_man.preload(TokenizedCluewebDoc, doc_ids)


def load_tf_multiple(doc_ids):
    return load_multiple(CluewebDocTF, doc_ids)


def save_tf(doc_id, tf):
    return save(CluewebDocTF, doc_id, tf)