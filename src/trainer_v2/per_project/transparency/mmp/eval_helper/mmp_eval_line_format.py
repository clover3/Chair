import json

from pytrec_eval import RelevanceEvaluator

from adhoc.eval_helper.pytrec_helper import eval_by_pytrec, convert_ranked_list_to_dict
from cpath import output_path
from adhoc.eval_helper.line_format_to_trec_ranked_list import \
    build_ranked_list_from_qid_pid_scores
from misc_lib import path_join, average

from dataset_specific.msmarco.passage.processed_resource_loader import load_msmarco_sub_samples_as_qd_pair, \
    get_dataset_quad_payload_path
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.eval_helper.eval_line_format import predict_qd_itr_save_score_lines, \
    batch_score_and_save_score_lines
from trec.qrel_parse import load_qrels_structured
from trec.trec_parse import load_ranked_list_grouped
from trec.types import TrecRankedListEntry

from typing import List, Iterable, Callable, Dict, Tuple


def eval_dev100_for_tune(dataset, run_name):
    metric = "recip_rank"
    scores_path = path_join(output_path, "lines_scores", "tune", f"{run_name}_{dataset}.txt")
    qid_pid_path = path_join("data", "msmarco", dataset, "corpus.tsv")
    judgment_path = path_join("data", "msmarco", "qrels.dev.tsv")
    ranked_list_path = path_join(output_path, "ranked_list", "tune", f"{run_name}_{dataset}.txt")
    c_log.debug("build_ranked_list_from_qid_pid_scores")
    build_ranked_list_from_qid_pid_scores(qid_pid_path, run_name, ranked_list_path, scores_path)
    ranked_list: Dict[str, List[TrecRankedListEntry]] = load_ranked_list_grouped(ranked_list_path)
    doc_scores = convert_ranked_list_to_dict(ranked_list)
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
    scores_path = get_line_scores_path(run_name, dataset)
    qid_pid_path = path_join("data", "msmarco", dataset, "corpus.tsv")
    return eval_from_score_lines_dev(dataset, metric, qid_pid_path, run_name, scores_path)


def get_line_scores_path(run_name, dataset):
    scores_path = path_join(output_path, "lines_scores", f"{run_name}_{dataset}.txt")
    return scores_path


def eval_test_ndcg(dataset, run_name):
    metric = "ndcg"
    scores_path = get_line_scores_path(run_name, dataset)
    qid_pid_path = get_dataset_quad_payload_path(dataset)
    return eval_from_score_lines_test(dataset, metric, qid_pid_path, run_name, scores_path)


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


def eval_from_score_lines_test(dataset, metric, qid_pid_path, run_name, scores_path):
    judgment_path = path_join("data", "msmarco", "passage", dataset, )
    return eval_from_score_lines_inner(dataset, metric, qid_pid_path, judgment_path, run_name, scores_path)


def eval_from_score_lines_inner(dataset, metric, qid_pid_path, judgment_path, run_name, scores_path):
    ranked_list_path = path_join(output_path, "ranked_list", f"{run_name}_{dataset}.txt")
    c_log.debug("build_ranked_list_from_qid_pid_scores")
    build_ranked_list_from_qid_pid_scores(qid_pid_path, run_name, ranked_list_path, scores_path)
    return eval_by_pytrec(judgment_path, ranked_list_path, metric)


def eval_from_score_lines_json_qrel(dataset, metric, qid_pid_path, judgment_path, run_name, scores_path):
    ranked_list_path = path_join(output_path, "ranked_list", f"{run_name}_{dataset}.txt")
    c_log.debug("build_ranked_list_from_qid_pid_scores")
    build_ranked_list_from_qid_pid_scores(qid_pid_path, run_name, ranked_list_path, scores_path)
    ranked_list: Dict[str, List[TrecRankedListEntry]] = load_ranked_list_grouped(ranked_list_path)
    doc_scores = convert_ranked_list_to_dict(ranked_list)
    c_log.debug("load_qrels_structured")
    qrels = json.load(open(judgment_path, "r"))
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
    itr = iter(load_msmarco_sub_samples_as_qd_pair(dataset))
    predict_and_save_scores_w_itr(score_fn, dataset, run_name, itr, data_size)


def predict_and_save_scores_w_itr(score_fn, dataset, run_name, itr, data_size):
    scores_path = path_join(output_path, "lines_scores", f"{run_name}_{dataset}.txt")
    predict_qd_itr_save_score_lines(score_fn, itr, scores_path, data_size)


def predict_and_batch_save_scores(
        score_fn: Callable[[List[Tuple[str, str]]], Iterable[float]],
        dataset: str,
        run_name: str,
        data_size=0,
):
    itr = iter(load_msmarco_sub_samples_as_qd_pair(dataset))
    max_batch_size = 1024
    scores_path = path_join(output_path, "lines_scores", f"{run_name}_{dataset}.txt")
    batch_score_and_save_score_lines(score_fn, itr, scores_path, data_size, max_batch_size)


def main():
    dataset = "dev_sample100"
    run_name = "bm25"
    print(eval_dev100_for_tune(dataset, run_name))


if __name__ == "__main__":
    main()

