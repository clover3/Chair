import argparse
import os
import sys
from typing import List, Dict, Tuple, Callable

import scipy.special
import scipy.special

from arg.qck.decl import get_format_handler, FormatHandler
from arg.qck.prediction_reader import load_combine_info_jsons
from arg.qck.qck_multi_save_to_trec_form import top_k_average
from arg.qck.save_to_trec_form import get_max_score_from_doc_parts
from arg.qck.trec_helper import score_d_to_trec_style_predictions
from cpath import output_path
from estimator_helper.output_reader import join_prediction_with_info
from list_lib import lmap
from misc_lib import exist_or_mkdir, group_by, average
from trec.trec_parse import write_trec_ranked_list_entry


def summarize_score(info: Dict,
                    prediction_file_path: str,
                    f_handler: FormatHandler,
                    combine_score: Callable,
                    score_type) -> Dict[Tuple[str, str], float]:
    key_logit = "logits"
    data: List[Dict] = join_prediction_with_info(prediction_file_path, info, ["data_id", key_logit])

    def logit_to_score_softmax(logit):
        return scipy.special.softmax(logit)[1]

    def get_score(entry):
        if score_type == "softmax":
            return logit_to_score_softmax(entry['logits'])
        elif score_type == "raw":
            return entry[key_logit][0]
        elif score_type == "scalar":
            return entry[key_logit]
        else:
            assert False

    grouped: Dict[Tuple[str, str], List[Dict]] = group_by(data, f_handler.get_pair_id)
    print("Group size:", len(grouped))
    out_d = {}
    for pair_id, items in grouped.items():
        items_before4 = [item for item in items if item['idx'] < 4]
        scores = lmap(get_score, items_before4)
        final_score = combine_score(scores)
        out_d[pair_id] = final_score

    num_items_per_group = average(lmap(len, grouped.values()))
    print("Num items per group : ", num_items_per_group)
    return out_d


def get_score_d(pred_file_path: str, info: Dict, f_handler: FormatHandler, combine_strategy: str, score_type: str):
    print("Reading from :", pred_file_path)
    if combine_strategy == "top_k":
        print("using top k")
        combine_score = top_k_average
    elif combine_strategy == "avg":
        combine_score = average
        print("using avg")
    elif combine_strategy == "max":
        print("using max")
        combine_score = max
    elif combine_strategy == "avg_then_doc_max":
        combine_score = average
        print("using avg then max")
    elif combine_strategy == "max_then_doc_max":
        combine_score = max
        print("using avg then max")
    else:
        assert False
    score_d = summarize_score(info, pred_file_path, f_handler, combine_score, score_type)

    if combine_strategy == "avg_then_doc_max" or combine_strategy == "max_then_doc_max":
        score_d = get_max_score_from_doc_parts(score_d)

    return score_d


def save_to_common_path(pred_file_path: str, info_file_path: str, run_name: str,
                        input_type: str,
                        max_entry: int,
                        combine_strategy: str,
                        score_type: str,
                        shuffle_sort: bool
                        ):
    f_handler = get_format_handler(input_type)
    info: Dict = load_combine_info_jsons(info_file_path, f_handler.get_mapping(), f_handler.drop_kdp())
    print("Info has {} entries".format(len(info)))
    score_d = get_score_d(pred_file_path, info, f_handler, combine_strategy, score_type)
    ranked_list = score_d_to_trec_style_predictions(score_d, run_name, max_entry, shuffle_sort)

    save_dir = os.path.join(output_path, "ranked_list")
    exist_or_mkdir(save_dir)
    save_path = os.path.join(save_dir, run_name + ".txt")
    write_trec_ranked_list_entry(ranked_list, save_path)
    print("Saved at : ", save_path)


parser = argparse.ArgumentParser(description='')
parser.add_argument("--pred_path")
parser.add_argument("--info_path")
parser.add_argument("--run_name")
parser.add_argument("--input_type", default="qck")
parser.add_argument("--max_entry", default=100)
parser.add_argument("--combine_strategy", default="avg_then_doc_max")
parser.add_argument("--score_type", default="softmax")
parser.add_argument("--shuffle_sort", default=False)


if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])
    save_to_common_path(args.pred_path,
                        args.info_path,
                        args.run_name,
                        args.input_type,
                        int(args.max_entry),
                        args.combine_strategy,
                        args.score_type,
                        args.shuffle_sort
    )