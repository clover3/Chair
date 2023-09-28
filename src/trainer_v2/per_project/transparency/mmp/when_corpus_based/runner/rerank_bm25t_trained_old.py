import logging

from dataset_specific.msmarco.passage.passage_resource_loader import load_msmarco_sub_samples_as_qd_pair
from table_lib import tsv_iter
from taskman_client.task_proxy import get_task_manager_proxy
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.when_corpus_based.when_bm25t import get_bm25t_when, get_bm25t_when_trained
from trainer_v2.per_project.transparency.mmp.eval_helper.eval_line_format import predict_and_save_scores, \
    predict_and_save_scores_w_itr, eval_on_train_when_0
from cpath import output_path
from misc_lib import path_join, select_third_fourth


def main():
    c_log.setLevel(logging.DEBUG)
    param_path = path_join(
        output_path, "msmarco", "passage", "when_trained")
    bm25t = get_bm25t_when_trained(param_path)
    run_name = "bm25t_when_trained"
    dataset = "train_when_0"
    n_item = 230958
    itr = load_msmarco_sub_samples_as_qd_pair(dataset)
    predict_and_save_scores_w_itr(bm25t.score, dataset, run_name, itr, n_item)
    score = eval_on_train_when_0(run_name)

    c_log.info("Mapping was used {} times".format(bm25t.n_mapping_used))
    print(f"MRR:\t{score}")
    proxy = get_task_manager_proxy()
    proxy.report_number(run_name, score, "when", "MRR")


if __name__ == "__main__":
    main()

