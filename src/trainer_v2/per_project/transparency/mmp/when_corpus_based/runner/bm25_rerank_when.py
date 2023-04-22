import logging

from adhoc.bm25_class import BM25
from cpath import at_output_dir
from dataset_specific.msmarco.passage.load_term_stats import load_msmarco_passage_term_stat
from dataset_specific.msmarco.passage.passage_resource_loader import tsv_iter, load_msmarco_sub_samples
from misc_lib import write_to_lines, TELI, select_third_fourth
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.eval_helper.eval_line_format import eval_dev100_mrr, \
    predict_and_save_scores_w_itr, eval_on_train_when_0
from cpath import output_path
from misc_lib import path_join


def get_bm25() -> BM25:
    cdf, df = load_msmarco_passage_term_stat()
    bm25 = BM25(df, cdf, 25, k1=0.1, k2=0, b=0.1)
    return bm25


def main():
    run_name = "bm25"
    bm25 = get_bm25()
    dataset = "train_when_0"
    n_item = 230958
    itr = load_msmarco_sub_samples(dataset)
    predict_and_save_scores_w_itr(bm25.score, dataset, run_name, itr, n_item)
    score = eval_on_train_when_0(run_name)
    print(f"MRR:\t{score}")


if __name__ == "__main__":
    main()