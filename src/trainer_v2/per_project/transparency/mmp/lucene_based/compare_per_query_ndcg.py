import sys
from typing import List, Iterable, Callable, Dict, Tuple, Set
from pytrec_eval import RelevanceEvaluator

from adhoc.eval_helper.pytrec_helper import load_qrels_as_structure_from_any
from misc_lib import average
from tab_print import print_table
from trec.trec_parse import load_ranked_list_grouped


def main():
    rl1_path = sys.argv[1]
    rl2_path = sys.argv[2]

    def load_ranked_list(rl_path):
        rlg = load_ranked_list_grouped(rl_path)
        score_d = {}
        for qid, entries in rlg.items():
            score_d[qid] = {e.doc_id: e.score for e in entries}
        return score_d

    def get_avg(score_per_query):
        scores = [score_per_query[qid][metric] for qid in score_per_query]
        return average(scores)

    judgment_path = sys.argv[3]
    qrels: Dict[str, Dict[str, int]] = load_qrels_as_structure_from_any(judgment_path)

    doc_scores1 = load_ranked_list(rl1_path)
    doc_scores2 = load_ranked_list(rl2_path)

    metric = "ndcg_cut_10"
    evaluator = RelevanceEvaluator(qrels, {metric})
    score_per_query1 = evaluator.evaluate(doc_scores1)
    score_per_query2 = evaluator.evaluate(doc_scores2)

    output = []
    for qid in score_per_query1:
        score1 = score_per_query1[qid][metric]
        score2 = score_per_query2[qid][metric]
        row = qid, score1, score2, score1 - score2
        output.append(row)

    output.sort(key=lambda x: x[3])
    print_table(output)

    print("Score for run 1", get_avg(score_per_query1))
    print("Score for run 2", get_avg(score_per_query2))


if __name__ == "__main__":
    main()