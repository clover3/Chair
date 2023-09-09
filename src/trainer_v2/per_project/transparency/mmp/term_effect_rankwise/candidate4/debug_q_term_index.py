import pickle
from collections import defaultdict
from typing import List, Iterable, Tuple

# Output (Doc_id, TFs Counter, base score)
from table_lib import tsv_iter
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.candidate4.select_q_n_gram import enum_n_gram
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.path_helper import get_qtfs_save_path, \
    get_grouped_queries_path
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.path_helper2 import MMPGAlignPathHelper, \
    get_cand4_path_helper
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.split_iter import get_valid_mmp_partition_for_train, \
    get_valid_mmp_partition
from misc_lib import path_join



def main():
    split = "train"
    config: MMPGAlignPathHelper = get_cand4_path_helper()
    qterm_index_dir = config.per_pair_candidates.q_term_index_path
    target_q_terms = "to see"

    def load_q_term_index(job_no):
        pickle_f = open(path_join(qterm_index_dir, str(job_no)), "rb")
        d = pickle.load(pickle_f)
        dd = defaultdict(list)
        dd.update(d)
        return dd

    inv_index = defaultdict(list)
    n_queries = 0
    n_appear = 0
    print("")
    for job_no in get_valid_mmp_partition(split):
        print(f"partition {job_no}")
        q_term_index = load_q_term_index(job_no)
        queries_path = get_grouped_queries_path(split, job_no)
        qid_queries = list(tsv_iter(queries_path))
        n_queries += len(qid_queries)
        n_appear_by_index = len(q_term_index[target_q_terms])
        print(n_appear_by_index)
        for qid, query in qid_queries:
            q_tokens = query.lower().split()
            for ngram in enum_n_gram(q_tokens, 2):
                s = " ".join(ngram)
                if s in [target_q_terms]:
                    inv_index[s].append(qid)
                    if not target_q_terms in query:
                        print("query: ", query)
                    n_appear += 1
    print(f"Target q_term {target_q_terms} appeared {n_appear} among {n_queries}")


if __name__ == "__main__":
    main()