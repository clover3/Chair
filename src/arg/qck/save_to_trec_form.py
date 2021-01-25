import json
import os
import sys
from typing import List, Dict, Tuple, Callable, NamedTuple

import scipy.special

from arg.qck.decl import get_qk_pair_id, get_qc_pair_id, get_format_handler, qck_convert_map, qk_convert_map, \
    qc_convert_map, FormatHandler
from arg.qck.prediction_reader import load_combine_info_jsons
from arg.qck.trec_helper import scrore_d_to_trec_style_predictions
from cpath import output_path
from estimator_helper.output_reader import join_prediction_with_info
from list_lib import lmap
from misc_lib import exist_or_mkdir, group_by, average
from trec.trec_parse import write_trec_ranked_list_entry


def top_k_average(items):
    k = 10
    items.sort(reverse=True)
    return average(items[:k])


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
        elif score_type == "tuple":
            return entry[key_logit][1]
        else:
            assert False

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
    elif input_type == "qckl":
        mapping = qck_convert_map
        group_fn = get_qc_pair_id
        drop_kdp = False
    else:
        assert False
    return drop_kdp, group_fn, mapping


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
    ranked_list = scrore_d_to_trec_style_predictions(score_d, run_name, max_entry, shuffle_sort)

    save_dir = os.path.join(output_path, "ranked_list")
    exist_or_mkdir(save_dir)
    save_path = os.path.join(save_dir, run_name + ".txt")
    write_trec_ranked_list_entry(ranked_list, save_path)
    print("Saved at : ", save_path)


def get_doc_id(doc_part_id: str):
    doc_id = "_".join(doc_part_id.split("_")[:-1])
    return doc_id


class DocPartScore(NamedTuple):
    query_id: str
    doc_id: str
    score: float

    def get_query_id_doc_id(self) -> Tuple[str, str]:
        return self.query_id, self.doc_id

    def get_score(self) -> float:
        return self.score


def get_max_score_from_doc_parts(score_d: Dict[Tuple[str, str], float]) -> Dict[Tuple[str, str], float]:
    dp_score_list = []
    for key, score in score_d.items():
        query_id, doc_part_id = key
        doc_id = get_doc_id(doc_part_id)
        dp_score = DocPartScore(query_id, doc_id, score)
        dp_score_list.append(dp_score)

    grouped = group_by(dp_score_list, DocPartScore.get_query_id_doc_id)
    out_d = {}
    for pair_id, seg_score_list in grouped.items():
        scores = lmap(DocPartScore.get_score, seg_score_list)
        score = max(scores)
        out_d[pair_id] = score
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


if __name__ == "__main__":
    run_config = json.load(open(sys.argv[1], "r"))
    save_to_common_path(run_config["prediction_path"],
                        run_config["info_path"],
                        run_config["run_name"],
                        run_config["input_type"],
                        run_config["max_entry"],
                        run_config["combine_strategy"],
                        run_config["score_type"],
                        False
    )