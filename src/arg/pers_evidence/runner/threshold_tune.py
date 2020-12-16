import sys
from typing import List, Dict

from data_generator.data_parser.robust2 import load_qrel
from misc_lib import average, get_f1
from trec.trec_parse import load_ranked_list_grouped, TrecRankedListEntry


def main():
    l1: Dict[str, List[TrecRankedListEntry]] = load_ranked_list_grouped(sys.argv[1])
    qrel: Dict[str, Dict[str, int]] = load_qrel(sys.argv[2])

    threshold_list = []
    ptr = 0.0
    for i in range(10):
        v = ptr + i/10
        threshold_list.append(v)
    res = []
    for t in threshold_list:
        prec_list = []
        recall_list = []
        for query, ranked_list in l1.items():
            gold_dict = qrel[query] if query in qrel else {}
            gold_docs = []
            for doc_id, label in gold_dict.items():
                if label:
                    gold_docs.append(doc_id)

            pred_list = []
            for e in ranked_list:
                if e.score > t:
                    pred_list.append(e.doc_id)

            common = set(gold_docs).intersection(set(pred_list))
            tp = len(common)
            prec = tp / len(pred_list) if pred_list else 1
            recall = tp / len(gold_docs) if gold_docs else 1
            prec_list.append(prec)
            recall_list.append(recall)

        prec = average(prec_list)
        recall = average(recall_list)
        f1 = get_f1(prec, recall)
        res.append((t, prec, recall, f1))

    for t, prec, recall, f1 in res:
        print(t, prec, recall, f1)



if __name__ == "__main__":
    main()