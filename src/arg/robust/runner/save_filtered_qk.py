from collections import Counter
from typing import List

from arg.qck.decl import QKUnit
from arg.qck.qk_summarize import QKOutEntry, get_score_from_logit
from arg.robust.visualizer.k_doc_viewer import load_qk_score
from cache import load_from_pickle, save_to_pickle
from exec_lib import run_func_with_config
from list_lib import lmap, left, lfilter


def main(config):
    qk_candidate: List[QKUnit] = load_from_pickle("robust_on_clueweb_qk_candidate")
    qk_out_entries: List[QKOutEntry] = load_qk_score(config)

    score_type = config['score_type']
    k = config['k']
    queries = left(qk_candidate)
    good_doc_list_d = {q.query_id: set() for q in queries}

    for entry in qk_out_entries:
        score = get_score_from_logit(score_type, entry.logits)
        if score > k:
            good_doc_list_d[entry.query.query_id].add(entry.kdp.doc_id)

    stat_count = Counter()
    def filter_map(qk_unit: QKUnit):
        query, kdp_list = qk_unit
        good_doc_list = good_doc_list_d[query.query_id]

        def is_good(kdp):
            return kdp.doc_id in good_doc_list

        new_kdp_list = lfilter(is_good, kdp_list)
        print("{} -> {}".format(len(kdp_list), len(new_kdp_list)))
        if not new_kdp_list:
            stat_count["no kdp"] += 1
        return query, new_kdp_list

    new_qk_candidate = lmap(filter_map, qk_candidate)
    print(stat_count)
    save_to_pickle(new_qk_candidate, "robust_on_clueweb_qk_candidate_filtered")


if __name__ == "__main__":
    run_func_with_config(main)