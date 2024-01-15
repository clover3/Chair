from adhoc.bm25_class import BM25
from dataset_specific.msmarco.passage.load_term_stats import load_msmarco_passage_term_stat
from taskman_client.task_proxy import get_task_manager_proxy
from trainer_v2.per_project.transparency.mmp.bm25t import BM25T
from trainer_v2.per_project.transparency.mmp.eval_helper.mmp_eval_line_format import predict_and_save_scores, \
    eval_dev_mrr


def run_dev_rerank_eval_with_bm25t(dataset, mapping, run_name):
    cdf, df = load_msmarco_passage_term_stat()
    bm25 = BM25(df, cdf, 25, 0.1, 100, 1.4)
    bm25t = BM25T(mapping, bm25.core)
    predict_and_save_scores(bm25t.score, dataset, run_name, 1000 * 1000)
    score = eval_dev_mrr(dataset, run_name)
    print(f"mrr:\t{score}")
    proxy = get_task_manager_proxy()
    proxy.report_number(run_name, score, dataset, "mrr")
    print(f"Recip_rank:\t{score}")
