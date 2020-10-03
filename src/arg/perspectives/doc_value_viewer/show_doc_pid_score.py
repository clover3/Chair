import json
import sys
from collections import Counter
from typing import List, Dict

from arg.perspectives.doc_value_viewer.show_doc_values import load_doc_score_prediction
from arg.perspectives.load import get_claim_perspective_id_dict
from arg.perspectives.runner_ppnc.show_score_stat import load_cppnc_score_and_baseline_and_group
from arg.qck.doc_value_calculator import doc_value
from tab_print import print_table


def doc_value_group(value):
    if value > 0.05:
        return "good"
    elif value < -0.05:
        return "bad"
    else:
        return "no effect"


def main():
    run_config = json.load(open(sys.argv[1], "r"))
    passage_score_path = run_config['passage_score_path']
    payload_name = run_config['payload_name']

    doc_scores: Dict[int, List[float]] = dict(load_doc_score_prediction(passage_score_path))
    baseline_cid_grouped, cid_grouped, claim_d = load_cppnc_score_and_baseline_and_group(payload_name)
    gold = get_claim_perspective_id_dict()

    g_counter = Counter()
    columns = ["pid doc pair", "good", "bad", "no effect", "no effect pid"]
    rows = [columns]
    record = []
    for cid, pid_entries in cid_grouped.items():
        baseline_pid_entries = baseline_cid_grouped[cid]

        baseline_score_d = {}
        for cpid, a_thing_array in baseline_pid_entries:
            _, pid = cpid
            assert len(a_thing_array) == 1
            score = a_thing_array[0]['score']
            baseline_score_d[pid] = score

        gold_pids = gold[cid]

        labels = []
        counter = Counter()
        for cpid, things in pid_entries:
            _, pid = cpid
            label = any([pid in pids for pids in gold_pids])
            labels.append(label)
            base_score = baseline_score_d[pid]

            any_effect = False
            try:
                for doc_idx, per_doc in enumerate(things):
                    score = per_doc['score']
                    value = doc_value(score, base_score, int(label))
                    qknc_score = doc_scores[cid][doc_idx]
                    if qknc_score < 0:
                        continue
                    value_type = doc_value_group(value)
                    counter[value_type] += 1
                    if value_type in [ "good", "bad"]:
                        record.append((cid, pid, doc_idx, value_type))
                    if value_type != "no effect":
                        any_effect = True
                    counter["pid doc pair"] += 1
                if not any_effect:
                    counter["no effect pid"] += 1
            except KeyError:
                print(cid, doc_idx, "not found")
                pass
        row = [cid] + list([counter[c] for c in columns])
        rows.append(row)

        for key, count in counter.items():
            g_counter[key] += count

    row = ["all"] + list([g_counter[c] for c in columns])
    rows.append(row)
    row = ["rate"] + list([g_counter[c]/g_counter["pid doc pair"] for c in columns])
    rows.append(row)
    print_table(rows)

    print_table(record)


if __name__ == "__main__":
    main()