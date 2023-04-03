from typing import List, Iterable, Callable, Dict, Tuple, Set

from pytrec_eval import RelevanceEvaluator

from cpath import output_path, at_output_dir
from dataset_specific.msmarco.passage.build_ranked_list import build_ranked_list_from_qid_pid_scores
from misc_lib import path_join, average, TELI

from dataset_specific.msmarco.passage.passage_resource_loader import tsv_iter, load_msmarco_sub_samples
from misc_lib import select_first_second
from runnable.trec.pytrec_eval_wrap import convert_ranked_list
from trainer_v2.chair_logging import c_log
from trec.qrel_parse import load_qrels_structured
from trec.trec_parse import load_ranked_list_grouped
from trec.types import TrecRankedListEntry


def eval_dev100_for_tune(dataset, run_name):
    metric = "ndcg"
    scores_path = path_join(output_path, "lines_scores", "tune", f"{run_name}_{dataset}.txt")
    qid_pid_path = path_join("data", "msmarco", dataset, "corpus.tsv")
    judgment_path = path_join("data", "msmarco", "qrels.dev.tsv")
    ranked_list_path = path_join(output_path, "ranked_list", "tune", f"{run_name}_{dataset}.txt")
    c_log.debug("build_ranked_list_from_qid_pid_scores")
    build_ranked_list_from_qid_pid_scores(qid_pid_path, run_name, ranked_list_path, scores_path)
    ranked_list: Dict[str, List[TrecRankedListEntry]] = load_ranked_list_grouped(ranked_list_path)
    doc_scores = convert_ranked_list(ranked_list)
    c_log.debug("load_qrels_structured")
    qrels = load_qrels_structured(judgment_path)
    evaluator = RelevanceEvaluator(qrels, {metric})
    score_per_query = evaluator.evaluate(doc_scores)
    scores = [score_per_query[qid][metric] for qid in score_per_query]
    return average(scores)


def predict_and_save_scores(score_fn: Callable[[str, str], float],
                            dataset: str,
                            run_name: str,
                            data_size=0,
                            ):
    itr = iter(load_msmarco_sub_samples(dataset))
    scores_path = path_join(output_path, "lines_scores", "tune", f"{run_name}_{dataset}.txt")
    f = open(scores_path, "w")
    if data_size:
        itr = TELI(itr, data_size)
    for q, d in itr:
        score = score_fn(q, d)
        f.write("{}\n".format(score))


def main():
    dataset = "dev_sample100"
    run_name = "bm25"
    print(eval_dev100_for_tune(dataset, run_name))
    return NotImplemented


if __name__ == "__main__":
    main()

