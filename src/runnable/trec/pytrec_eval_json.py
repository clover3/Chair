import json
import sys
from misc_lib import average
from pytrec_eval import RelevanceEvaluator


def main():
    judgment_path = sys.argv[1]
    ranked_list_path = sys.argv[2]
    metric = sys.argv[3]

    qrels = json.load(open(judgment_path, "r"))
    doc_scores = json.load(open(ranked_list_path, "r"))
    evaluator = RelevanceEvaluator(qrels, {metric})
    score_per_query = evaluator.evaluate(doc_scores)
    print(score_per_query)

    scores = [score_per_query[qid][metric] for qid in score_per_query]
    print("{}\t{}".format(metric, average(scores)))


if __name__ == "__main__":
    main()
