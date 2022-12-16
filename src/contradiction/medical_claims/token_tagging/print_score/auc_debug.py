from typing import List, Iterable, Callable, Dict, Tuple, Set

from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_curve, auc

from contradiction.medical_claims.cont_classification.path_helper import load_raw_predictions
from contradiction.medical_claims.token_tagging.acc_eval.path_helper import load_sbl_binary_label
from contradiction.medical_claims.token_tagging.path_helper import get_binary_save_path_w_opt, get_save_path2
from contradiction.medical_claims.token_tagging.print_score.auc import load_run
from contradiction.token_tagging.acc_eval.defs import SentTokenLabel
from contradiction.token_tagging.acc_eval.eval_codes import calc_prec_rec_acc
from contradiction.token_tagging.acc_eval.parser import load_sent_token_binary_predictions
from list_lib import index_by_fn
from misc_lib import average, str_float_list
from tab_print import print_table
from trec.trec_parse import load_ranked_list_grouped
from trec.types import TrecRankedListEntry


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
    return prec_list, recall_list

from typing import List, Iterable, Callable, Dict, Tuple, Set


def main():
    tag = "mismatch"
    labels: List[SentTokenLabel] = load_sbl_binary_label(tag, "val")
    rlg1 = load_run("nlits86", tag)
    rlg2 = load_run("exact_match", tag)

    prec_list1, recall_list1 = eval_micro_auc(rlg1, labels)
    prec_list2, recall_list2 = eval_micro_auc(rlg2, labels)

    fig, ax = plt.subplots()
    ax.plot(recall_list1, prec_list1, color='blue')
    ax.plot(recall_list2, prec_list2, color='red')

    ax.set_title('Precision-Recall Curve')
    ax.set_ylabel('Precision')
    ax.set_xlabel('Recall')

    #display plot
    plt.show()


if __name__ == "__main__":
    main()