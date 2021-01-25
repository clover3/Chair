from collections import Counter
from typing import List, Dict, Tuple

from numpy import argmax

from cache import load_from_pickle
from data_generator.data_parser.robust2 import load_robust_qrel
from list_lib import right
from misc_lib import group_by, get_first, get_second


def main():
    score_d: Dict[Tuple[str, str, int], float] = load_from_pickle("robust_score_d")
    score_d2: Dict[Tuple[str, str, int], float] = load_from_pickle("robust_score_d2")

    qrel: Dict[str, Dict[str, int]] = load_robust_qrel()
    query_grouped = group_by(score_d.keys(), get_first)


    counter = Counter()
    for query_id in query_grouped:
        keys: List[Tuple[str, str, int]] = query_grouped[query_id]

        doc_id_grouped = group_by(keys, get_second)

        qrel_part = qrel[query_id] if query_id in qrel else {}
        for doc_id in doc_id_grouped:
            label: int = qrel_part[doc_id] if doc_id in qrel_part else 0
            cur_keys: List[Tuple[str, str, int]] = doc_id_grouped[doc_id]
            if len(cur_keys) == 1:
                continue
            summary = []
            summary2 = []
            for key in cur_keys:
                query_id2, doc_id2, passage_idx = key
                assert query_id2 == query_id
                assert doc_id2 == doc_id
                score = score_d[key]
                score2 = score_d2[key]
                summary.append((passage_idx, score))
                summary2.append((passage_idx, score2))

            summary.sort(key=get_first)
            summary2.sort(key=get_first)

            max_idx = int(argmax(right(summary)))
            max_idx2 = int(argmax(right(summary2)))

            if label:
                if max_idx == max_idx2:
                    counter[1] += 1
                else:
                    counter[0] += 1

    print(counter)
    accuracy = counter[1] / (counter[0] + counter[1])
    print("accuracy {}".format(accuracy))


if __name__ == "__main__":
    main()