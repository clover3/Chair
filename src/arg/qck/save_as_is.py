import json
import os
import sys
from typing import List, Iterable, Dict, Tuple

import scipy.special

from arg.qck.decl import get_format_handler, FormatHandler
from arg.qck.prediction_reader import load_combine_info_jsons
from cpath import output_path
from estimator_helper.output_reader import join_prediction_with_info
from list_lib import lmap
from misc_lib import group_by, exist_or_mkdir
from trec.trec_parse import write_trec_ranked_list_entry
from trec.types import TrecRankedListEntry


def summarize_score(info: Dict,
                    prediction_file_path: str,
                    f_handler: FormatHandler,
                    score_type) -> Iterable[TrecRankedListEntry]:
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
        elif score_type == "tuple":
            return entry[key_logit][1]
        else:
            assert False

    grouped: Dict[Tuple[str, str], List[Dict]] = group_by(data, f_handler.get_pair_id)
    print("Group size:", len(grouped))
    out_d = {}
    for pair_id, items in grouped.items():
        scores = lmap(get_score, items)
        query_id, doc_id = pair_id
        out_d[pair_id] = scores
        for score in scores:
            yield TrecRankedListEntry(query_id, doc_id, 0, score, "")



def save_to_common_path(pred_file_path: str, info_file_path: str, run_name: str,
                        input_type: str,
                        max_entry: int,
                        score_type: str,
                        shuffle_sort: bool
                        ):
    f_handler = get_format_handler(input_type)
    info: Dict = load_combine_info_jsons(info_file_path, f_handler.get_mapping(), f_handler.drop_kdp())
    print("Info has {} entries".format(len(info)))
    ranked_list = summarize_score(info, pred_file_path, f_handler, score_type)
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
                        run_config["score_type"],
                        False
    )