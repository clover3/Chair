from typing import List, Dict, Tuple

from numpy import argmax

from cache import load_from_pickle
from data_generator.data_parser.robust2 import load_robust_qrel
from list_lib import right
from misc_lib import group_by, get_first, get_second
from tab_print import print_table


def main():
    score_d: Dict[Tuple[str, str, int], float] = load_from_pickle("robust_score_d2")

    qrel: Dict[str, Dict[str, int]] = load_robust_qrel()
    query_grouped = group_by(score_d.keys(), get_first)

    for query_id in query_grouped:
        keys: List[Tuple[str, str, int]] = query_grouped[query_id]

        doc_id_grouped = group_by(keys, get_second)

        qrel_part = qrel[query_id] if query_id in qrel else {}
        pos_rows = []
        neg_rows = []
        for doc_id in doc_id_grouped:
            label: int = qrel_part[doc_id] if doc_id in qrel_part else 0
            cur_keys: List[Tuple[str, str, int]] = doc_id_grouped[doc_id]
            summary = []
            for key in cur_keys:
                query_id2, doc_id2, passage_idx = key
                assert query_id2 == query_id
                assert doc_id2 == doc_id
                score = score_d[key]
                summary.append((passage_idx, score))

            summary.sort(key=get_first)

            max_idx = int(argmax(right(summary)))

            score_str = list(["{0:.5f}".format(s) for s in right(summary)])

            max_passage_idx = summary[max_idx][0]
            row = [str(max_passage_idx)] + score_str
            if label:
                pos_rows.append(row)
            else:
                neg_rows.append(row)

        print(query_id)
        print("Positive")
        print_table(pos_rows)
        print("Negative")
        print_table(neg_rows[:30])


if __name__ == "__main__":
    main()