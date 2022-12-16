from typing import List, Iterable, Callable, Dict, Tuple, Set

from sklearn.metrics import precision_recall_curve, auc

from contradiction.medical_claims.cont_classification.path_helper import load_raw_predictions
from contradiction.medical_claims.token_tagging.acc_eval.path_helper import load_sbl_binary_label
from contradiction.medical_claims.token_tagging.path_helper import get_binary_save_path_w_opt, get_save_path2
from contradiction.token_tagging.acc_eval.defs import SentTokenLabel
from contradiction.token_tagging.acc_eval.eval_codes import calc_prec_rec_acc
from contradiction.token_tagging.acc_eval.parser import load_sent_token_binary_predictions
from list_lib import index_by_fn
from misc_lib import average, str_float_list
from tab_print import print_table
from trec.trec_parse import load_ranked_list_grouped
from trec.types import TrecRankedListEntry


def load_run(run_name, tag_type) -> Dict[str, List[TrecRankedListEntry]]:
    rl_path = get_save_path2(run_name, tag_type)
    rlg = load_ranked_list_grouped(rl_path)
    return rlg


def eval_avg_auc(
        rlg: Dict[str, List[TrecRankedListEntry]],
        labels: List[SentTokenLabel], adjust_to_prediction_length=False
    ) -> float:
    labels_d: Dict[str, SentTokenLabel] = index_by_fn(lambda x: x.qid, labels)

    auc_v_list = []
    for qid in rlg:
        entries: List[TrecRankedListEntry] = rlg[qid]
        try:
            labels_for_qid: List[int] = labels_d[qid].labels
            doc_id_to_score = {}
            for e in entries:
                doc_id_to_score[e.doc_id] = e.score

            predictions: List[float] = [doc_id_to_score[str(i)] for i in range(len(labels_for_qid))]
            assert len(labels_for_qid) == len(predictions)
            prec_list, recall_list, _ = precision_recall_curve(labels_for_qid, predictions)
            v = auc(recall_list, prec_list)
            auc_v_list.append(v)
        except KeyError:
            pass

    if len(auc_v_list) != len(labels) and not adjust_to_prediction_length:
        raise IndexError("Ranked list is missing preditions {} -> {}".format(len(labels), len(auc_v_list)))

    return average(auc_v_list)


def eval_micro_auc(
        rlg: Dict[str, List[TrecRankedListEntry]],
        labels: List[SentTokenLabel], adjust_to_prediction_length=False
    ) -> float:
    labels_d: Dict[str, SentTokenLabel] = index_by_fn(lambda x: x.qid, labels)

    all_labels = []
    all_predictions = []
    for qid in rlg:
        entries: List[TrecRankedListEntry] = rlg[qid]
        try:
            labels_for_qid: List[int] = labels_d[qid].labels
            doc_id_to_score = {}
            for e in entries:
                doc_id_to_score[e.doc_id] = e.score

            predictions: List[float] = [doc_id_to_score[str(i)] for i in range(len(labels_for_qid))]
            assert len(labels_for_qid) == len(predictions)
            all_labels.extend(labels_for_qid)
            all_predictions.extend(predictions)
        except KeyError:
            pass

    prec_list, recall_list, threshold_list = precision_recall_curve(all_labels, all_predictions)
    print(str_float_list(prec_list))
    print(str_float_list(recall_list))
    print(str_float_list(threshold_list))
    v = auc(recall_list, prec_list)
    return v


def show_for_mismatch():
    mismatch_run_list = ["random", "nlits86", "psearch", "coattention", "senli", "deletion", "exact_match"]
    conflict_run_list = ["random", "nlits86", "psearch", "senli", "deletion", "exact_match"]
    tag = "mismatch"
    mismatch_run_list = ["nlits86", "exact_match"]
    three_digit = "{0:.3f}".format
    labels: List[SentTokenLabel] = load_sbl_binary_label(tag, "val")

    run_list = {
        'mismatch': mismatch_run_list,
        'conflict': conflict_run_list,
    }[tag]
    table = []
    for run_name in run_list:
        try:
            print(run_name)
            rlg = load_run(run_name, tag)
            s = eval_micro_auc(rlg, labels)
            table.append([run_name, three_digit(s)])
        except FileNotFoundError as e:
            print(e)
    print_table(table)


if __name__ == "__main__":
    show_for_mismatch()