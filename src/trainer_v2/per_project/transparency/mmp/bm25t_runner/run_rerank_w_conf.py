import sys
from typing import Iterable, Tuple

from omegaconf import OmegaConf

from adhoc.bm25_class import BM25
from dataset_specific.msmarco.passage.load_term_stats import load_msmarco_passage_term_stat
from misc_lib import select_third_fourth
from table_lib import tsv_iter
from trainer_v2.per_project.transparency.mmp.bm25t import BM25T
from trainer_v2.per_project.transparency.mmp.bm25t_helper import load_mapping_from_align_scores
from trainer_v2.per_project.transparency.mmp.eval_helper.eval_line_format import predict_qd_itr_save_score_lines


def main():
    conf_path = sys.argv[1]

    conf = OmegaConf.load(conf_path)
    table_path = conf.table_path
    cut = 0.1
    mapping_val = 0.1
    quad_tsv_path = conf.rerank_payload_path
    mapping = load_mapping_from_align_scores(table_path, cut, mapping_val)

    cdf, df = load_msmarco_passage_term_stat()
    bm25 = BM25(df, cdf, 25, 0.1, 100, 1.4)
    bm25t = BM25T(mapping, bm25.core)
    qd_iter: Iterable[Tuple[str, str]] = select_third_fourth(tsv_iter(quad_tsv_path))
    predict_qd_itr_save_score_lines(
        bm25t.score,
        qd_iter,
        conf.score_save_path,
        conf.data_size)


if __name__ == "__main__":
    main()
