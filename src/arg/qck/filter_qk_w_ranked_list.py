from typing import List, Dict

from arg.qck.decl import QKUnit, KDP
from cache import load_from_pickle
from list_lib import lfilter, right, lmap
from trec.trec_parse import load_ranked_list_grouped, TrecRankedListEntry


def filter_with_ranked_list(qk_untis : List[QKUnit],
                            ranked_list_d: Dict[str, List[TrecRankedListEntry]],
                            threshold,
                            top_k,
                            ) -> List[QKUnit]:

    out_qk_units = []
    for q, k_list in qk_untis:
        try:
            cur_ranked_list = ranked_list_d[q.query_id]
            entries: Dict[str, TrecRankedListEntry] = {e.doc_id: e for e in cur_ranked_list}
            n_k_list = len(k_list)

            not_found_set = set()
            def get_score(k: KDP):
                key = k.to_str()
                if key in entries:
                    s: TrecRankedListEntry = entries[key]
                    return s.score
                else:
                    not_found_set.add(key)
                    return -1e10

            k_list.sort(key=get_score, reverse=True)

            def higher(k: KDP) -> bool:
                return get_score(k) >= threshold

            if threshold is not None:
                k_list = lfilter(higher, k_list)

            if top_k is None or top_k == -1:
                pass
            else:
                k_list = k_list[:top_k]
            out_qk_units.append((q, k_list))
            if not_found_set:
                print("For query {}, {} of {} do not have score".format(q.query_id, len(not_found_set), n_k_list))
        except KeyError as e:
            print(e, "KeyError", q.query_id)

    print(lmap(len, right(out_qk_units)))
    return out_qk_units


def filter_with_ranked_list_path(qk_name: str,
                                 ranked_list_path: str,
                                 threshold,
                                 top_k):
    rlg: Dict[str, List[TrecRankedListEntry]] = load_ranked_list_grouped(ranked_list_path)
    qk_units = load_from_pickle(qk_name)
    new_qk_units = filter_with_ranked_list(qk_units, rlg, threshold, top_k)
    return new_qk_units

