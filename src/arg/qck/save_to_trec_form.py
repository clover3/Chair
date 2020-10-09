import argparse
import json
import os
import sys

import scipy.special

from arg.qck.decl import get_qk_pair_id, get_qc_pair_id, get_format_handler, qck_convert_map, qk_convert_map, \
    qc_convert_map
from arg.qck.prediction_reader import load_combine_info_jsons
from arg.qck.trec_helper import scrore_d_to_trec_style_predictions
from estimator_helper.output_reader import join_prediction_with_info
from list_lib import lmap
from misc_lib import exist_or_mkdir, group_by, average

parser = argparse.ArgumentParser(description='')


parser.add_argument("--pred_path")
parser.add_argument("--info_path")
parser.add_argument("--run_name")
parser.add_argument("--input_type", default="qck")
parser.add_argument("--max_entry", default=-1)


from cpath import output_path
from evals.trec import write_trec_ranked_list_entry
from typing import List, Dict, Tuple


def top_k_average(items):
    k = 10
    items.sort(reverse=True)
    return average(items[:k])


def summarize_score(info_dir, prediction_file, input_type, combine_strategy) -> Dict[Tuple[str, str], float]:
    f_handler = get_format_handler(input_type)
    if combine_strategy == "top_k":
        print("using top k")
        combine_score = top_k_average
    elif combine_strategy == "avg":
        combine_score = average
        print("using avg")
    elif combine_strategy == "max":
        print("using max")
        combine_score = max
    else:
        assert False

    info = load_combine_info_jsons(info_dir, f_handler.get_mapping(), f_handler.drop_kdp())
    print("Info has {} entries".format(len(info)))
    data: List[Dict] = join_prediction_with_info(prediction_file, info, ["data_id", "logits"])

    def logit_to_score_softmax(logit):
        return scipy.special.softmax(logit)[1]

    def get_score(entry):
        return logit_to_score_softmax(entry['logits'])

    grouped: Dict[Tuple[str, str], List[Dict]] = group_by(data, f_handler.get_pair_id)
    print("Group size:", len(grouped))
    out_d = {}
    for pair_id, items in grouped.items():
        scores = lmap(get_score, items)
        final_score = combine_score(scores)
        out_d[pair_id] = final_score

    num_items_per_group = average(lmap(len, grouped.values()))
    print("Num items per group : ", num_items_per_group)
    return out_d


def get_mapping_per_input_type(input_type):
    if input_type == "qck":
        mapping = qck_convert_map
        group_fn = get_qc_pair_id
        drop_kdp = True
    elif input_type == "qc":
        mapping = qc_convert_map
        group_fn = get_qc_pair_id
        drop_kdp = False
    elif input_type == "qk":
        mapping = qk_convert_map
        group_fn = get_qk_pair_id
        drop_kdp = True
    else:
        assert False
    return drop_kdp, group_fn, mapping


def save_to_common_path(pred_file_path, info_file_path, run_name, input_type, max_entry, combine_strategy):
    print("Reading from :", pred_file_path)
    score_d = summarize_score(info_file_path, pred_file_path, input_type, combine_strategy)
    ranked_list = scrore_d_to_trec_style_predictions(score_d, run_name, max_entry)

    save_dir = os.path.join(output_path, "ranked_list")
    exist_or_mkdir(save_dir)
    save_path = os.path.join(save_dir, run_name + ".txt")
    write_trec_ranked_list_entry(ranked_list, save_path)
    print("Saved at : ", save_path)


if __name__ == "__main__":
    run_config = json.load(open(sys.argv[1], "r"))
    save_to_common_path(run_config["prediction_path"],
                        run_config["info_path"],
                        run_config["run_name"],
                        run_config["input_type"],
                        run_config["max_entry"],
                        run_config["combine_strategy"]
    )