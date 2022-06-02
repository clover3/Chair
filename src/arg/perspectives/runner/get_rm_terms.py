import json
import os

from arg.perspectives.pc_run_path import get_train_query_file
from cpath import project_root
from dataset_specific.clue_path import index_name_list
from galagos.query_to_all_clueweb_disk import get_rm_terms
from list_lib import lmap, flatten


# query is made from query_gen.py


def work():
    query_files = lmap(get_train_query_file, range(0, 122))

    def read(query_file):
        j = json.load(open(query_file))
        return j['queries']

    all_query = flatten(lmap(read, query_files))
    disk_name = index_name_list[0]

    out_root = "/mnt/nfs/work3/youngwookim/data/perspective/train_claim_perspective/rm3"
    base_head = os.path.join(project_root, "script", "rm3", "head.sh")
    base_tail = os.path.join(project_root, "script", "rm3", "tail.sh")
##
    idx = 200
    window = 100
    while idx < len(all_query):
        sh_path = os.path.join(project_root, "script", "rm3", "{}.sh".format(idx))
        queries = all_query[idx:idx+window]
        get_rm_terms(queries, disk_name, out_root, base_head, base_tail, sh_path)
        idx += window

if __name__ == "__main__":
    work()

