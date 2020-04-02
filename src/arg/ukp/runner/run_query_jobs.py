from arg.ukp.ukp_q_path import get_ranked_list_save_dir, get_all_query_file_names
from galagos import query_to_all_clueweb_disk
from galagos.query_runs_ids import Q_CONFIG_ID_BM25_UKP
from misc_lib import exist_or_mkdir
from sydney_clueweb.clue_path import index_name_list


# query is made from query_gen.py



def work():
    ranked_list_save_root = get_ranked_list_save_dir(Q_CONFIG_ID_BM25_UKP)
    exist_or_mkdir(ranked_list_save_root)
    query_files = get_all_query_file_names(Q_CONFIG_ID_BM25_UKP)
    query_to_all_clueweb_disk.send(query_files,
                                   index_name_list[:1],
                                   "ukp_{}".format(Q_CONFIG_ID_BM25_UKP),
                                   ranked_list_save_root)


if __name__ == "__main__":
    work()
