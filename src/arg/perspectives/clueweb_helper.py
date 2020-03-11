import os

from datastore.interface import load
from datastore.table_names import TokenizedCluewebDoc
from galagos.basic import load_galago_ranked_list


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
