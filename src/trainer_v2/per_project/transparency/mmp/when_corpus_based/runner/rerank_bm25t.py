import logging

from dataset_specific.msmarco.passage.passage_resource_loader import tsv_iter, load_msmarco_sub_samples
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.when_corpus_based.when_bm25t import get_bm25t_when, get_bm25t_when2
from trainer_v2.per_project.transparency.mmp.eval_helper.eval_line_format import predict_and_save_scores, \
    predict_and_save_scores_w_itr, eval_train_when_0
from cpath import output_path
from misc_lib import path_join, select_third_fourth


def main():
    c_log.setLevel(logging.DEBUG)
    bm25t = get_bm25t_when()
    run_name = "bm25t_when"
    dataset = "train_when_0"
    n_item = 230958
    itr = load_msmarco_sub_samples(dataset)
    predict_and_save_scores_w_itr(bm25t.score, dataset, run_name, itr, n_item)
    score = eval_train_when_0(run_name)

    c_log.info("Mapping was used {} times".format(bm25t.n_mapping_used))
    print(f"MRR:\t{score}")


def main():
    c_log.setLevel(logging.DEBUG)
    bm25t = get_bm25t_when2()
    run_name = "bm25t_when2"
    dataset = "train_when_0"
    n_item = 230958
    itr = load_msmarco_sub_samples(dataset)
    predict_and_save_scores_w_itr(bm25t.score, dataset, run_name, itr, n_item)
    score = eval_train_when_0(run_name)

    c_log.info("Mapping was used {} times".format(bm25t.n_mapping_used))
    print(f"MRR:\t{score}")



if __name__ == "__main__":
    main()

