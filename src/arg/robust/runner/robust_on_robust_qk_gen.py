from typing import List, Dict

from arg.qck.decl import KDP_BT, QKUnitBT
from arg.robust.qc_common import to_qck_queries
from cache import save_to_pickle
from data_generator.data_parser.robust import load_robust04_title_query
from galagos.parse import load_galago_ranked_list
from galagos.types import SimpleRankedListEntry
from misc_lib import TimeEstimator
from tlm.robust.load import load_robust_tokens_for_predict


def config1():
    return {
        'step_size': 300,
        'window_size': 300,
        'top_n': 10,
    }


class KDPGeneratorFirstOnly:
    def __init__(self, q_res_path, top_n, window_size):
        self.robust_tokens: Dict[str, List[str]] = load_robust_tokens_for_predict()
        self.ranked_list: Dict[str, List[SimpleRankedListEntry]] = load_galago_ranked_list(q_res_path)
        self.top_n = top_n
        self.window_size = window_size

    def get_kdps(self, query_id: str) -> List[KDP_BT]:
        kdp_list = []
        for e in self.ranked_list[query_id][:self.top_n]:
            doc_tokens = self.robust_tokens[e.doc_id]
            tokens = doc_tokens[0: self.window_size]
            kdp = KDP_BT(e.doc_id, 0, 0, tokens)
            kdp_list.append(kdp)
        return kdp_list


def get_candidates(q_res_path, top_n) -> List[QKUnitBT]:
    queries: Dict[str, str] = load_robust04_title_query()
    print("{} queries".format(len(queries)))
    qck_queries = to_qck_queries(queries)
    kdp_gen = KDPGeneratorFirstOnly(q_res_path, top_n, 300)
    print("Making queries")
    all_candidate = []
    ticker = TimeEstimator(len(queries))
    for q in qck_queries:
        ticker.tick()
        try:
            doc_part_list = kdp_gen.get_kdps(q.query_id)
            e = q, doc_part_list
            all_candidate.append(e)
        except KeyError as e:
            print(e)
    return all_candidate


def main():
    q_res_path = "/mnt/nfs/work3/youngwookim/data/robust_on_robust/robust_ho651_all.txt"
    save_name = "robust_qk_candidate_651_top10"
    top_n = 10
    all_candidate = get_candidates(q_res_path, top_n)
    save_to_pickle(all_candidate, save_name)


if __name__ == "__main__":
    main()
