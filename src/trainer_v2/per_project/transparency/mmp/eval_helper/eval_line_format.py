from typing import List, Callable, Dict, Tuple

from pytrec_eval import RelevanceEvaluator

from cpath import output_path
from dataset_specific.msmarco.passage.runner.build_ranked_list import build_ranked_list_from_qid_pid_scores
from list_lib import apply_batch
from misc_lib import path_join, average, TELI, ceil_divide, TimeEstimator

from dataset_specific.msmarco.passage.passage_resource_loader import load_msmarco_sub_samples
from runnable.trec.pytrec_eval_wrap import convert_ranked_list
from trainer_v2.chair_logging import c_log
from trec.qrel_parse import load_qrels_structured
from trec.trec_parse import load_ranked_list_grouped
from trec.types import TrecRankedListEntry

from typing import List, Iterable, Callable, Dict, Tuple, Set


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
    c_log.info("%d queries", len(score_per_query))
    scores = [score_per_query[qid][metric] for qid in score_per_query]
    return average(scores)


def eval_dev_mrr(dataset, run_name):
    metric = "recip_rank"
    scores_path = path_join(output_path, "lines_scores", f"{run_name}_{dataset}.txt")
    qid_pid_path = path_join("data", "msmarco", dataset, "corpus.tsv")
    return eval_from_score_lines_dev(dataset, metric, qid_pid_path, run_name, scores_path)


def eval_dev_ndcg(dataset, run_name):
    metric = "ndcg"
    scores_path = path_join(output_path, "lines_scores", f"{run_name}_{dataset}.txt")
    qid_pid_path = path_join("data", "msmarco", dataset, "corpus.tsv")
    return eval_from_score_lines_dev(dataset, metric, qid_pid_path, run_name, scores_path)


def eval_on_train_when_0(run_name):
    dataset = "train_when_0"
    metric = "recip_rank"
    scores_path = path_join(output_path, "lines_scores", f"{run_name}_{dataset}.txt")
    qid_pid_path = path_join(output_path, "msmarco", "passage", "when_full", "0")
    return eval_from_score_lines_train(dataset, metric, qid_pid_path, run_name, scores_path)


def eval_from_score_lines_dev(dataset, metric, qid_pid_path, run_name, scores_path):
    judgment_path = path_join("data", "msmarco", "qrels.dev.tsv")
    return eval_from_score_lines_inner(dataset, metric, qid_pid_path, judgment_path, run_name, scores_path)


def eval_from_score_lines_train(dataset, metric, qid_pid_path, run_name, scores_path):
    judgment_path = path_join("data", "msmarco", "qrels.train.tsv")
    return eval_from_score_lines_inner(dataset, metric, qid_pid_path, judgment_path, run_name, scores_path)


def eval_from_score_lines_inner(dataset, metric, qid_pid_path, judgment_path, run_name, scores_path):
    ranked_list_path = path_join(output_path, "ranked_list", f"{run_name}_{dataset}.txt")
    c_log.debug("build_ranked_list_from_qid_pid_scores")
    build_ranked_list_from_qid_pid_scores(qid_pid_path, run_name, ranked_list_path, scores_path)
    ranked_list: Dict[str, List[TrecRankedListEntry]] = load_ranked_list_grouped(ranked_list_path)
    doc_scores = convert_ranked_list(ranked_list)
    c_log.debug("load_qrels_structured")
    qrels = load_qrels_structured(judgment_path)
    evaluator = RelevanceEvaluator(qrels, {metric})
    score_per_query = evaluator.evaluate(doc_scores)
    c_log.debug("Computed scores for %d queries", len(score_per_query))
    scores = [score_per_query[qid][metric] for qid in score_per_query]
    return average(scores)


def predict_and_save_scores(score_fn: Callable[[str, str], float],
                            dataset: str,
                            run_name: str,
                            data_size=0,
                            ):
    itr = iter(load_msmarco_sub_samples(dataset))
    predict_and_save_scores_w_itr(score_fn, dataset, run_name, itr, data_size)


def predict_and_save_scores_w_itr(score_fn, dataset, run_name, itr, data_size):
    scores_path = path_join(output_path, "lines_scores", f"{run_name}_{dataset}.txt")
    f = open(scores_path, "w")
    if data_size:
        itr = TELI(itr, data_size)
    for q, d in itr:
        score = score_fn(q, d)
        f.write("{}\n".format(score))


def predict_and_batch_save_scores(
        score_fn: Callable[[List[Tuple[str, str]]], Iterable[float]],
        dataset: str,
        run_name: str,
        data_size=0,
):
    itr = iter(load_msmarco_sub_samples(dataset))
    max_batch_size = 1024
    scores_path = path_join(output_path, "lines_scores", f"{run_name}_{dataset}.txt")
    score_and_save_score_lines(itr, score_fn, scores_path, max_batch_size, data_size)


def score_and_save_score_lines(itr, score_fn, scores_path, max_batch_size, data_size):
    f = open(scores_path, "w")
    if data_size:
        n_batch = ceil_divide(data_size, max_batch_size)
        ticker = TimeEstimator(n_batch)
    for batch in apply_batch(itr, max_batch_size):
        scores: Iterable[float] = score_fn(batch)
        for x, s in zip(batch, scores):
            f.write("{}\n".format(s))

        if data_size:
            ticker.tick()


def main():
    dataset = "dev_sample100"
    run_name = "bm25"
    print(eval_dev100_for_tune(dataset, run_name))


if __name__ == "__main__":
    main()

