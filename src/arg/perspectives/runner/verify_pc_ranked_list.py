

import os

from galagos import query_to_all_clueweb_disk
from sydney_clueweb.clue_path import index_name_list


# query is made from query_gen.py

def work():
    query_dir = "/mnt/nfs/work3/youngwookim/code/Chair/output/perspective_train_claim_perspective_query"
    query_files = list([os.path.join(query_dir, "{}.json".format(i)) for i in range(0, 122)])
    out_root = '/mnt/nfs/work3/youngwookim/data/perspective/train_claim_perspective/all_query_results'
    query_to_all_clueweb_disk.verify_result(query_files,
                                              index_name_list[1:],
                                              out_root,
                                              )

if __name__ == "__main__":
    work()


