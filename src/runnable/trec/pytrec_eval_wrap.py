import sys
from typing import List, Dict

from evals.metrics import get_metric_fn
from misc_lib import average
from trec.qrel_parse import load_qrels_structured
from trec.trec_parse import load_ranked_list_grouped
from trec.types import TrecRankedListEntry
from pytrec_eval import RelevanceEvaluator


def convert_ranked_list(ranked_list: Dict[str, List[TrecRankedListEntry]]):
    out_d = {}
    for qid, entries in ranked_list.items():
        per_q = {}
        for e in entries:
            per_q[e.doc_id] = e.score
        out_d[qid] = per_q
    return out_d




def main():
    judgment_path = sys.argv[1]
    ranked_list_path = sys.argv[2]
    metric = sys.argv[3]

    qrels = load_qrels_structured(judgment_path)
    ranked_list: Dict[str, List[TrecRankedListEntry]] = load_ranked_list_grouped(ranked_list_path)
    doc_scores = convert_ranked_list(ranked_list)
    evaluator = RelevanceEvaluator(qrels, {metric})
    score_per_query = evaluator.evaluate(doc_scores)

    scores = [score_per_query[qid][metric] for qid in score_per_query]
    print("{}\t{}".format(metric, average(scores)))


if __name__ == "__main__":
    main()