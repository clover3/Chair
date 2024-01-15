import json
import sys
from typing import List, Dict

from pytrec_eval import RelevanceEvaluator
from adhoc.eval_helper.pytrec_helper import convert_ranked_list_to_dict

from misc_lib import average
from trec.trec_parse import load_ranked_list_grouped
from trec.types import TrecRankedListEntry


def main():
    judgment_path = sys.argv[1]
    ranked_list_path = sys.argv[2]
    metric = sys.argv[3]

    qrels = json.load(open(judgment_path, "r"))

    if ranked_list_path.endswith(".json"):
        doc_scores = json.load(open(ranked_list_path, "r"))
    else:
        ranked_list: Dict[str, List[TrecRankedListEntry]] = load_ranked_list_grouped(ranked_list_path)
        doc_scores = convert_ranked_list_to_dict(ranked_list)

    evaluator = RelevanceEvaluator(qrels, {metric})
    score_per_query = evaluator.evaluate(doc_scores)
    print(score_per_query)

    scores = [score_per_query[qid][metric] for qid in score_per_query]
    print("{}\t{}".format(metric, average(scores)))


if __name__ == "__main__":
    main()
