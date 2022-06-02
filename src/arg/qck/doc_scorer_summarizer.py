import argparse
import json
import sys
from typing import List, Iterable, Dict

from arg.qck.decl import QKUnit, qk_convert_map
from arg.qck.prediction_reader import load_combine_info_jsons
from cache import save_to_pickle
from list_lib import lmap, lfilter
from misc_lib import group_by, average
from tlm.estimator_output_reader import join_prediction_with_info

parser = argparse.ArgumentParser(description='File should be stored in ')
parser.add_argument("--prediction_path")
parser.add_argument("--info_path")
parser.add_argument("--config_path")
parser.add_argument("--save_name")


def get_regression_score(entry):
    return entry['logits'][0]


def extract_qk_unit(info_path, pred_path, config_path) -> Iterable[QKUnit]:
    info = load_combine_info_jsons(info_path, qk_convert_map, False)
    predictions = join_prediction_with_info(pred_path, info)
    grouped: Dict[str, List[Dict]] = group_by(predictions, lambda x: x['query'].query_id)
    config = json.load(open(config_path, "r"))
    score_cut = config['score_cut']
    top_k = config['top_k']

    def is_good(entry):
        return get_regression_score(entry) > score_cut

    select_rate_list = []
    qk_units = []
    for qid, entries in grouped.items():
        any_entry = entries[0]
        query = any_entry['query']
        good_entries = lfilter(is_good, entries)
        good_entries.sort(key=get_regression_score, reverse=True)
        selected_entries = good_entries[:top_k]
        if not selected_entries:
            continue
        kd_list = lmap(lambda x: x['kdp'], selected_entries)
        qk_units.append((query, kd_list))

        select_rate = len(selected_entries) / len(entries)
        select_rate_list.append(select_rate)

    print("{} of {} qk units selected".format(len(qk_units), len(grouped)))
    print("average select rate", average(select_rate_list))
    return qk_units


def main():
    args = parser.parse_args(sys.argv[1:])
    r = extract_qk_unit(args.info_path, args.prediction_path, args.config_path)
    save_to_pickle(list(r), args.save_name)


if __name__ == "__main__":
    main()
