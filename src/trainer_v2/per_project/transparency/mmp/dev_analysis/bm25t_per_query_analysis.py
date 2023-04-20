from typing import List, Iterable, Callable, Dict, Tuple, Set

from pytrec_eval import RelevanceEvaluator

from cpath import output_path
from dataset_specific.msmarco.passage.runner.build_ranked_list import build_ranked_list_from_qid_pid_scores
from misc_lib import path_join, two_digit_float
from runnable.trec.pytrec_eval_wrap import convert_ranked_list
from trec.qrel_parse import load_qrels_structured
from trec.trec_parse import load_ranked_list_grouped
from trec.types import TrecRankedListEntry


def get_scores(run_name, dataset):
    judgment_path = path_join("data", "msmarco", "qrels.train.tsv")
    metric = "recip_rank"
    scores_path = path_join(output_path, "lines_scores", f"{run_name}_{dataset}.txt")
    qid_pid_path = path_join(output_path, "msmarco", "passage", "when_full", "0")
    ranked_list_path = path_join(output_path, "ranked_list", f"{run_name}_{dataset}.txt")
    build_ranked_list_from_qid_pid_scores(qid_pid_path, run_name, ranked_list_path, scores_path)
    ranked_list: Dict[str, List[TrecRankedListEntry]] = load_ranked_list_grouped(ranked_list_path)
    doc_scores = convert_ranked_list(ranked_list)
    qrels = load_qrels_structured(judgment_path)
    evaluator = RelevanceEvaluator(qrels, {metric})
    score_per_query = evaluator.evaluate(doc_scores)
    scores = [(qid, score_per_query[qid][metric]) for qid in score_per_query]
    return scores


def main():
    run_name1 = "bm25"
    run_name2 = "bm25t_when"
    dataset = "train_when_0"
    scores1 = get_scores(run_name1, dataset)
    scores2 = get_scores(run_name2, dataset)

    for a1, a2 in zip(scores1, scores2):
        qid1, s1 = a1
        qid2, s2 = a2
        print(str(qid1) + "\t" + "\t".join(map(two_digit_float, [s1, s2, s2-s1])))



if __name__ == "__main__":
    main()