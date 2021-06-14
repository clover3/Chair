import argparse
import random
import sys
from typing import List, Dict, Tuple

import scipy.special

from arg.qck.decl import get_format_handler
from arg.qck.prediction_reader import load_combine_info_jsons
from estimator_helper.output_reader import join_prediction_with_info
from misc_lib import group_by, tprint, find_max_idx, SuccessCounter


def save_to_common_path(pred_file_path: str, info_file_path: str, run_name: str,
                        input_type: str,
                        max_entry: int,
                        combine_strategy: str,
                        score_type: str,
                        shuffle_sort: bool
                        ):
    f_handler = get_format_handler(input_type)
    info: Dict = load_combine_info_jsons(info_file_path, f_handler.get_mapping(), f_handler.drop_kdp())
    key_logit = "logits"
    data: List[Dict] = join_prediction_with_info(pred_file_path, info, ["data_id", key_logit])

    def logit_to_score_softmax(logit):
        return scipy.special.softmax(logit)[1]

    def get_score(entry):
        if score_type == "softmax":
            return logit_to_score_softmax(entry['logits'])
        elif score_type == "raw":
            return entry[key_logit][0]
        elif score_type == "scalar":
            return entry[key_logit]
        elif score_type == "tuple":
            return entry[key_logit][1]
        else:
            assert False

    grouped: Dict[Tuple[str, str], List[Dict]] = group_by(data, f_handler.get_pair_id)
    tprint("Group size:", len(grouped))
    out_d = {}
    sc = SuccessCounter()
    sc_random = SuccessCounter()

    for pair_id, items in grouped.items():
        label_idx = []
        max_score_idx = find_max_idx(items, get_score)
        max_item = items[max_score_idx]
        for item in items:
            if item['label']:
                label_idx.append(item['passage_idx'])

        if len(items) > 1:
            random_idx = random.randint(0, len(items)-1)
            if max_item['passage_idx'] in label_idx:
                sc.suc()
            else:
                sc.fail()

            if random_idx in label_idx:
                sc_random.suc()
            else:
                sc_random.fail()

        # print("{} {} {}".format(max_item['passage_idx'], label_idx, len(items)))

    print(sc.get_total(), sc.get_suc_prob())
    print('random', sc_random.get_total(), sc_random.get_suc_prob())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='File should be stored in ')
    parser.add_argument("--pred_path")
    parser.add_argument("--info_path")
    parser.add_argument("--run_name")
    parser.add_argument("--input_type", default="qck")
    parser.add_argument("--max_entry", default=100)
    parser.add_argument("--combine_strategy", default="avg_then_doc_max")
    parser.add_argument("--score_type", default="softmax")
    parser.add_argument("--shuffle_sort", default=False)

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