# TODO
# get query split
# enumerate (query-doc) candidate
#
import os

from base_type import FilePath
from cpath import output_path, pjoin
from galagos.parse import load_galago_ranked_list, write_ranked_list_from_d
from galagos.types import RankedListDict
from misc_lib import exist_or_mkdir
from tlm.alt_emb.clef_ehealth.qrel import load_clef_qrels
from tlm.alt_emb.clef_ehealth.split_query import get_query_split


def main():
    train_queries, test_queries = get_query_split()
    out_dir = pjoin(output_path, "eHealth")
    exist_or_mkdir(out_dir)
    ranked_list_path = FilePath("/mnt/nfs/work3/youngwookim/data/CLEF_eHealth_working/ranked_list_filtered")
    ranked_list: RankedListDict = load_galago_ranked_list(ranked_list_path)
    qrels = load_clef_qrels()

    new_d = {}
    for query in test_queries:
        new_d[query.qid] = ranked_list[query.qid]

    save_path = os.path.join(out_dir, 'test_baseline.list')
    write_ranked_list_from_d(new_d, save_path)


if __name__ == "__main__":
    main()
