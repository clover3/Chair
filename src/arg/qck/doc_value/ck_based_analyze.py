import json
import sys
from typing import List, Dict, Tuple

from arg.qck.decl import get_format_handler, get_qk_pair_id
from arg.qck.prediction_reader import load_combine_info_jsons
from arg.qck.qk_summarize import get_score_from_logit
from arg.qck.save_to_trec_form import get_doc_id
from cache import load_cache
from evals.basic_func import get_acc_prec_recall
from list_lib import dict_value_map
from misc_lib import group_by
from tab_print import print_table
from tlm.estimator_output_reader import join_prediction_with_info
from trec.trec_parse import load_qrels_flat


def prec_recall(pred_file_path: str, info_file_path: str, input_type: str, score_type: str,
         qrel_path: str):
    judgments_raw: Dict[str, List[Tuple[str, int]]] = load_qrels_flat(qrel_path)
    judgments = dict_value_map(dict, judgments_raw)

    grouped = load_cache("ck_based_analysis")
    key_logit = "logits"

    if grouped is None:
        f_handler = get_format_handler(input_type)
        info: Dict = load_combine_info_jsons(info_file_path, f_handler.get_mapping(), f_handler.drop_kdp())
        data: List[Dict] = join_prediction_with_info(pred_file_path, info, ["data_id", key_logit])
        grouped = group_by(data, get_qk_pair_id)

    def get_score(entry):
        return get_score_from_logit(score_type, entry[key_logit])

    def get_label(query_id, candidate_id):
        judge_dict = judgments[query_id]
        if candidate_id in judge_dict:
            return judge_dict[candidate_id]
        else:
            return 0

    head = ["query_id", "kdp_id",
            "accuracy", "precision", "recall", "f1",
            "tp", "fp", "tn", "fn"]
    rows = [head]
    for pair_id, items in grouped.items():
        query_id, kdp_id = pair_id
        if query_id not in judgments:
            continue

        e_list: List[Tuple[str, float]] = []

        labels = []
        predictions = []
        for item in items:
            score = get_score(item)
            doc_part_id = item['candidate'].id
            doc_id = get_doc_id(doc_part_id)
            e = (doc_id, score)
            e_list.append(e)
            label = bool(get_label(query_id, doc_id))
            labels.append(label)
            prediction = score > 0.5
            predictions.append(prediction)

        scores = get_acc_prec_recall(predictions, labels)

        row = [query_id, kdp_id,
               scores['accuracy'], scores['precision'], scores['recall'], scores['f1'],
               scores['tp'], scores['fp'], scores['tn'], scores['fn']]
        rows.append(row)
    print_table(rows)



def show_tp(pred_file_path: str, info_file_path: str, input_type: str, score_type: str, qrel_path: str):
    judgments_raw: Dict[str, List[Tuple[str, int]]] = load_qrels_flat(qrel_path)
    judgments = dict_value_map(dict, judgments_raw)
    key_logit = "logits"

    def get_score(entry):
        return get_score_from_logit(score_type, entry[key_logit])

    def get_label(query_id, candidate_id):
        judge_dict = judgments[query_id]
        if candidate_id in judge_dict:
            return judge_dict[candidate_id]
        else:
            return 0

    rows = []
    grouped = load_cache("ck_based_analysis")
    for pair_id, items in grouped.items():
        query_id, kdp_id = pair_id
        if query_id not in judgments:
            continue

        e_list: List[Tuple[str, float]] = []
        n_rel = 0
        for item in items:
            score = get_score(item)
            doc_part_id = item['candidate'].id
            doc_id = get_doc_id(doc_part_id)
            e = (doc_id, score)
            e_list.append(e)

            label = bool(get_label(query_id, doc_id))
            if label:
                if score > 0.1:
                    row = [query_id, kdp_id, doc_part_id, score]
                    rows.append(row)
                n_rel += 1
        row = [len(items), n_rel]
        rows.append(row)

    print_table(rows)


def main(pred_file_path: str, info_file_path: str, input_type: str, score_type: str, qrel_path: str):
    show_tp(pred_file_path, info_file_path, input_type, score_type, qrel_path)


if __name__ == "__main__":
    run_config = json.load(open(sys.argv[1], "r"))
    main(run_config["prediction_dir"],
         run_config["info_path"],
         run_config["input_type"],
         run_config["score_type"],
         run_config["qrel_path"],
         )

