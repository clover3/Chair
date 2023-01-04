from typing import List, Iterable, Callable, Dict, Tuple, Set

from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_curve, auc

from contradiction.medical_claims.annotation_1.load_data import load_reviews_for_split
from contradiction.medical_claims.load_corpus import Review
from trainer_v2.per_project.tli.bioclaim_qa.path_helper import get_retrieval_save_path
from misc_lib import str_float_list
from tab_print import print_table
from trec.trec_parse import load_ranked_list_grouped
from trec.types import TrecRankedListEntry


def eval_micro_auc(
        rlg: Dict[str, List[TrecRankedListEntry]],
        get_label: Callable[[str, str], int]
    ) -> Tuple[List[float], List[float]]:
    all_labels: List[int] = []
    all_predictions = []
    for qid in rlg:
        entries: List[TrecRankedListEntry] = rlg[qid]
        for e in entries:
            label = get_label(qid, e.doc_id)
            all_labels.append(label)
            all_predictions.append(e.score)

    prec_list, recall_list, threshold_list = precision_recall_curve(all_labels, all_predictions)
    # print(str_float_list(prec_list))
    # print(str_float_list(recall_list))
    # print(str_float_list(threshold_list))
    return prec_list, recall_list


def get_label_d(split) -> Dict[Tuple[str, str], int]:
    review_list: List[Tuple[int, Review]] = load_reviews_for_split(split)
    d = {}
    for group_no, r in review_list:
        qid = str(group_no)
        for c in r.claim_list:
            doc_id: str = c.pmid
            d[qid, doc_id] = 1
    return d


def do_eval(split, run_name):
    rlg = load_ranked_list_grouped(get_retrieval_save_path(run_name))
    label_d = get_label_d(split)
    all_qid = set()
    for qid, _ in label_d:
        all_qid.add(qid)


    def get_label(qid, doc_id):
        if not qid in all_qid:
            raise KeyError

        if (qid, doc_id) in label_d:
            return label_d[qid, doc_id]
        else:
            return 0

    r = eval_micro_auc(rlg, get_label)
    return r


def main():
    run_list = ["bm25_test", "test_nli", "test_nli_pep", "test_nli_pep_idf", ]

    fig, ax = plt.subplots()

    color_d = {
        "bm25_test": 'red',
        "test_nli": "blue",
        "test_nli_pep": "gray",
        "test_nli_pep_idf": "yellow",
    }

    table = []
    for run_name in run_list:
        print(run_name)
        prec_list, recall_list = do_eval("test", run_name)
        r = auc(recall_list, prec_list)
        ax.plot(recall_list, prec_list, color=color_d[run_name])
        table.append((run_name, r))

    print_table(table)

    ax.set_title('Precision-Recall Curve')
    ax.set_ylabel('Precision')
    ax.set_xlabel('Recall')
    plt.show()


if __name__ == "__main__":
    main()