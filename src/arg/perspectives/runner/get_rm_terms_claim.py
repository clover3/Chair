import os

from arg.perspectives.load import load_train_claim_ids, load_dev_claim_ids, get_claims_from_ids
from arg.perspectives.query.query_gen import get_claims_query
from cpath import project_root
from galagos.query_to_all_clueweb_disk import get_rm_terms
from misc_lib import exist_or_mkdir
from sydney_clueweb.clue_path import index_name_list


# query is made from query_gen.py
def submit_rm_jobs(all_query, out_root):
    disk_name = index_name_list[0]

    base_head = os.path.join(project_root, "script", "rm3", "head.sh")
    base_tail = os.path.join(project_root, "script", "rm3", "tail.sh")
##
    idx = 10
    window = 10
    while idx < len(all_query):
        sh_path = os.path.join(project_root, "script", "rm3", "{}.sh".format(idx))
        queries = all_query[idx:idx+window]
        get_rm_terms(queries, disk_name, out_root, base_head, base_tail, sh_path)
        idx += window


def work():
    claim_ids, split_name = (load_train_claim_ids(), "train")
    print("Num claims in train : ", len(list(claim_ids)))

    exit()
    def submit_jobs_inner(claim_ids, split_name):
        claims = get_claims_from_ids(claim_ids)
        queries = get_claims_query(claims)
        out_root = "/mnt/nfs/work3/youngwookim/data/perspective/{}_claim_rm3".format(split_name)
        exist_or_mkdir(out_root)
        submit_rm_jobs(queries, out_root)

    claim_ids, split_name = (load_dev_claim_ids(), "dev")
    submit_jobs_inner(claim_ids, split_name)


if __name__ == "__main__":
    work()

