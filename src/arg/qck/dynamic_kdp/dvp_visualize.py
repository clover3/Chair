import pickle
import sys
from collections import Counter
from typing import List, Dict

from arg.qck.doc_value_calculator import DocValueParts
from misc_lib import group_by, get_second
from tab_print import print_table


def main(dvp_pickle_path):
    dvp: List[DocValueParts] = pickle.load(open(dvp_pickle_path, "rb"))

    def get_qid(e: DocValueParts):
        return e.query.query_id
    # Group by doc id
    dvp_qid_grouped: Dict[str, List[DocValueParts]] = group_by(dvp, get_qid)

    def get_doc_id_idx(e: DocValueParts):
        return e.kdp.doc_id, e.kdp.passage_idx

    def get_doc_id(e: DocValueParts):
        return e.kdp.doc_id

    value_types = ["good", "bad", "none"]
    head = ["qid"] + value_types
    rows = [head]

    rows2 = []

    for qid, entries in dvp_qid_grouped.items():
        # Q : How many kdp are useful?
        # Q : Does relevant matter?
        kdp_grouped = group_by(entries, get_doc_id_idx)
        counter = Counter()
        doc_value = Counter()
        for kdp_id, entries2 in kdp_grouped.items():
            doc_id, _  = kdp_id
            value_avg: float = sum([e.value for e in entries2])
            if value_avg > 1:
                counter["good"] += 1
                doc_value[doc_id] += 1
            elif value_avg < -1:
                counter["bad"] += 1
                doc_value[doc_id] -= 1
            else:
                counter["none"] += 1

        row = [qid] + [counter[k] for k in value_types]
        rows.append(row)

        doc_value_list = list(doc_value.items())
        doc_value_list.sort(key=get_second, reverse=True)
        rows2.append([qid])
        for doc_id, value in doc_value_list[:10]:
            if value > 0:
                rows2.append([doc_id, value])
        doc_value_list.sort(key=get_second)
        for doc_id, value in doc_value_list[:10]:
            if value < 0:
                rows2.append([doc_id, value])

    print_table(rows)
    print_table(rows2)


if __name__ == "__main__":
    main(sys.argv[1])
