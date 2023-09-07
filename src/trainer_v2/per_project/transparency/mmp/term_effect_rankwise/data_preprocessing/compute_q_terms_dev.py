import pickle
import sys
from collections import Counter
from typing import List, Iterable, Tuple

from adhoc.bm25_class import BM25
from dataset_specific.msmarco.passage.passage_resource_loader import FourItem, enum_grouped
from table_lib import tsv_iter
from dataset_specific.msmarco.passage.path_helper import get_mmp_grouped_sorted_path

# Output (Doc_id, TFs Counter, base score)
from trainer_v2.per_project.transparency.mmp.bm25_paramed import get_bm25_mmp_25_01_01
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.data_preprocessing.compute_q_terms import work_for
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.path_helper import get_qtfs_save_path


def main():
    split = "dev"
    for job_no in range(110, 111):
        print(job_no)
        itr = tsv_iter(get_mmp_grouped_sorted_path(split, job_no))
        g_itr: Iterable[List[FourItem]] = enum_grouped(itr)
        bm25 = get_bm25_mmp_25_01_01()
        qid_qtf_entries = work_for(g_itr, bm25)
        save_path = get_qtfs_save_path(split, job_no)
        pickle.dump(qid_qtf_entries, open(save_path, "wb"))


if __name__ == "__main__":
    main()
