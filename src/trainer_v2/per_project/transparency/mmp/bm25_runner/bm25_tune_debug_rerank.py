from adhoc.bm25_class import BM25
from adhoc.ks_tokenizer import KrovetzSpaceTokenizer
from dataset_specific.msmarco.passage.doc_indexing.index_path_helper import get_bm25_sp_stem_resource_path_helper
from adhoc.other.bm25_retriever_helper import load_bm25_resources, get_bm25_stats_from_conf
from dataset_specific.msmarco.passage.load_term_stats import load_msmarco_passage_term_stat
from dataset_specific.msmarco.passage.processed_resource_loader import load_msmarco_sub_samples_as_qd_pair
from trainer_v2.per_project.transparency.mmp.eval_helper.eval_line_format import predict_qd_itr_save_score_lines
from trainer_v2.per_project.transparency.mmp.eval_helper.mmp_eval_line_format import eval_dev100_for_tune
from typing import List, Iterable, Callable, Dict, Tuple, Set
from cpath import output_path
from misc_lib import path_join


def get_bm25(bm25_params: Dict[str, float]) -> BM25:
    cdf, df = load_msmarco_passage_term_stat()
    bm25 = BM25(df, cdf, bm25_params['dl'],
                bm25_params['k1'],
                bm25_params['k2'],
                bm25_params['b'],
                )
    return bm25


def get_bm25_sp_stem(bm25_params):
    conf = get_bm25_sp_stem_resource_path_helper()
    avdl = 52
    avdl, cdf, df, dl = get_bm25_stats_from_conf(conf, avdl)
    bm25 = BM25(df, cdf, bm25_params['dl'],
                bm25_params['k1'],
                bm25_params['k2'],
                bm25_params['b'],
                )
    tokenizer = KrovetzSpaceTokenizer()
    bm25.tokenizer = tokenizer
    return bm25

def run_eval_common(bm25, run_name):
    dataset = "dev_sample100"
    score_fn = bm25.score
    itr = iter(load_msmarco_sub_samples_as_qd_pair(dataset))
    data_size = 100 * 100
    scores_path = path_join(output_path, "lines_scores", "tune", f"{run_name}_{dataset}.txt")
    predict_qd_itr_save_score_lines(score_fn, itr, scores_path, data_size)

    score = eval_dev100_for_tune(dataset, run_name)
    print(f"{run_name}\t{score}")
    return score


def run_eval_on_bm25(bm25_params) -> float:
    bm25 = get_bm25_sp_stem(bm25_params)
    run_name = "bm25_7"
    print(run_name)
    score = run_eval_common(bm25, run_name)
    return score


def main():
    bm25_params = {
        "k1": 0.1,
        "k2": 100,
        "b": 1.4,
        "dl": 26,
    }
    run_eval_on_bm25(bm25_params)


if __name__ == "__main__":
    main()