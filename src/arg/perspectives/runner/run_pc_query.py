from arg.perspectives.pc_run_path import get_query_file
from galagos import query_to_all_clueweb_disk
from list_lib import lmap
from misc_lib import exist_or_mkdir
from sydney_clueweb.clue_path import index_name_list

# query is made from query_gen.py

ranked_list_save_root = '/mnt/nfs/work3/youngwookim/data/perspective/train_claim_perspective/bm25_k0_q_res'


def work():
    exist_or_mkdir(ranked_list_save_root)
    query_files = lmap(get_query_file, range(0, 122))
    query_to_all_clueweb_disk.send(query_files,
                                   index_name_list[1:],
                                   "perspective_train_claim_perspective_query_k0",
                                   ranked_list_save_root)


if __name__ == "__main__":
    work()
