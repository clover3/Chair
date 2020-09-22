import os
import sys
from typing import List, Dict, Tuple

from arg.perspectives.load import get_claims_from_ids, load_dev_claim_ids, get_claim_perspective_id_dict
from arg.perspectives.ppnc.get_doc_value import load_and_group_predictions
from arg.perspectives.types import CPIDPair
from cpath import output_path
from list_lib import lmap, foreach
from misc_lib import exist_or_mkdir, average, BinHistogram, bool_to_yn
from tab_print import print_table


def load_data():
    save_name = sys.argv[1]
    out_dir = os.path.join(output_path, "cppnc")
    exist_or_mkdir(out_dir)
    info_file_path = os.path.join(out_dir, save_name + ".info")
    pred_file_path = os.path.join(out_dir, save_name + ".score")
    d_ids = list(load_dev_claim_ids())
    claims: List[Dict] = get_claims_from_ids(d_ids)
    claim_d = {c['cId']: c['text'] for c in claims}
    cid_grouped = load_and_group_predictions(info_file_path, pred_file_path)
    save_name = "baseline_cppnc"
    baseline_info_file_path = os.path.join(out_dir, save_name + ".info")
    baseline_pred_file_path = os.path.join(out_dir, save_name + ".score")
    baseline_cid_grouped = load_and_group_predictions(baseline_info_file_path, baseline_pred_file_path)
    return baseline_cid_grouped, cid_grouped, claim_d


def main():
    baseline_cid_grouped, cid_grouped, claim_d = load_data()
    gold = get_claim_perspective_id_dict()

    bin_keys = ["< 0.05", "< 0.50", "< 0.95", "< 1"]

    def bin_fn(item: float):
        if item > 0.95:
            return "< 1"
        elif item > 0.5:
            return "< 0.95"
        elif item > 0.05:
            return "< 0.50"
        else:
            return "< 0.05"

    for cid, pid_entries in cid_grouped.items():
        baseline_pid_entries = baseline_cid_grouped[cid]

        baseline_score_d = {}
        for cpid, a_thing_array in baseline_pid_entries:
            _, pid = cpid
            assert len(a_thing_array) == 1
            score = a_thing_array[0]['score']
            baseline_score_d[pid] = score

        gold_pids = gold[cid]

        def get_score_per_pid_entry(p_entries: Tuple[CPIDPair, List[Dict]]):
            cpid, entries = p_entries
            return average(lmap(lambda e: e['score'], entries))

        pid_entries.sort(key=get_score_per_pid_entry, reverse=True)

        s = "{} : {}".format(cid, claim_d[cid])
        print(s)
        head_row = [""] + bin_keys
        rows = [head_row]
        for cpid, things in pid_entries:
            histogram = BinHistogram(bin_fn)
            _, pid = cpid
            label = any([pid in pids for pids in gold_pids])
            label_str = bool_to_yn(label)
            base_score = baseline_score_d[pid]
            base_score_str = "{0:.2f}".format(base_score)
            scores: List[float] = lmap(lambda x: (x['score']), things)
            foreach(histogram.add, scores)
            row = [label_str, base_score_str] + [str(histogram.counter[bin_key]) for bin_key in bin_keys]
            rows.append(row)
        print_table(rows)


if __name__ == "__main__":
    main()
