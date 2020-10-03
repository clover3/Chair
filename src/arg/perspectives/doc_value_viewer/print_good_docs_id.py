from collections import Counter
from typing import List, Dict

from arg.perspectives.doc_value_viewer.show_doc_value2 import fetch_score_per_pid, get_score_from_entry
from arg.perspectives.load import get_claim_perspective_id_dict
from arg.perspectives.ppnc.get_doc_value import load_cppnc_score, load_baseline
from arg.qck.doc_value_calculator import doc_value
from list_lib import left
from tab_print import print_table


def doc_value_group(value):
    if value > 0.05:
        return "good"
    elif value < -0.05:
        return "bad"
    else:
        return "no effect"


def main():
    save_name = "qcknc_val"
    cid_grouped: Dict[str, Dict[str, List[Dict]]] = load_cppnc_score(save_name)
    baseline_cid_grouped: Dict[int, List] = load_baseline("train_baseline")

    # baseline_cid_grouped, cid_grouped, claim_d = load_cppnc_related_data()
    gold = get_claim_perspective_id_dict()

    columns = ["cid", "doc_id", "num_good-num_bad"]
    rows = [columns]
    for cid_s, pid_entries in cid_grouped.items():
        cid = int(cid_s)
        baseline_pid_entries = baseline_cid_grouped[cid]
        baseline_score_d: Dict[int, float] = fetch_score_per_pid(baseline_pid_entries)

        gold_pids = gold[cid]

        labels = []
        per_doc_counter = Counter()
        for pid, entries in pid_entries.items():
            label = any([pid in pids for pids in gold_pids])
            labels.append(label)
            base_score = baseline_score_d[int(pid)]

            try:
                for doc_idx, entry in enumerate(entries):
                    doc_id = entry['kdp'].doc_id
                    score = get_score_from_entry(entry)
                    value = doc_value(score, base_score, int(label))
                    value_type = doc_value_group(value)
                    per_doc_counter[doc_id, value_type] += 1

            except KeyError:
                print(cid, doc_idx, "not found")
                pass
        doc_ids = set(left(per_doc_counter.keys()))
        for doc_id in doc_ids:
            n_good = per_doc_counter[doc_id, "good"]
            n_bad = per_doc_counter[doc_id, "bad"]
            doc_score = n_good - n_bad
            row = [cid, doc_id, doc_score]
            if doc_score > 2 or doc_score < -2:
                rows.append(row)

    print_table(rows)


if __name__ == "__main__":
    main()