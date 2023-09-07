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
    freq_q_terms: List[str] = config.load_qterm_candidates()
    freq_q_terms_set = set(freq_q_terms)

    inv_index = defaultdict(list)
    for job_no in get_valid_mmp_partition(split):
        queries_path = get_grouped_queries_path(split, job_no)
        for qid, query in tsv_iter(queries_path):
            q_tokens = query.lower().split()
            for ngram in enum_n_gram(q_tokens, 2):
                s = " ".join(ngram)
                if s in freq_q_terms_set:
                    inv_index[s].append(qid)

        save_path = path_join(config.per_pair_candidates.q_term_index_path, str(job_no))
        print(save_path)
        pickle.dump(dict(inv_index), open(save_path, "wb"))


if __name__ == "__main__":
    main()







if __name__ == "__main__":
    main()