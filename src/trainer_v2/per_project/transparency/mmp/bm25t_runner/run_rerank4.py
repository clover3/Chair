import sys
from typing import Dict

from omegaconf import OmegaConf

from adhoc.other.bm25_retriever_helper import get_tokenize_fn, get_bm25_stats_from_conf, build_bm25_scoring_fn_from_conf
from taskman_client.wrapper3 import JobContext
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.bm25t_4 import BM25T_4
from trainer_v2.per_project.transparency.mmp.table_readers import load_align_scores
from trainer_v2.per_project.transparency.mmp.eval_helper.rerank_w_conf import run_rerank_with_conf_common
from trainer_v2.per_project.transparency.mmp.parallel_helper import parallel_run


def get_bm25t_scorer_fn(conf):
    value_mapping: Dict[str, Dict[str, float]] = load_align_scores(conf.table_path)
    bm25_conf = OmegaConf.load(conf.bm25conf_path)
    tokenize_fn = get_tokenize_fn(bm25_conf)
    avdl, cdf, df, dl = get_bm25_stats_from_conf(bm25_conf, None)
    scoring_fn = build_bm25_scoring_fn_from_conf(bm25_conf, avdl, cdf)

    bm25t = BM25T_4(value_mapping, scoring_fn, tokenize_fn, df)

    def parallel_score_fn(qd_list):
        n_per_job = 1000 * 100
        if len(qd_list) > n_per_job:
            split_n = min(len(qd_list) // n_per_job, 12)
        else:
            split_n = 1
        if split_n > 1:
            ret = parallel_run(qd_list, bm25t.score_batch, split_n)
        else:
            ret = bm25t.score_batch(qd_list)
        return ret

    return parallel_score_fn


def main():
    c_log.info(__file__)
    get_scorer_fn = get_bm25t_scorer_fn
    conf_path = sys.argv[1]
    conf = OmegaConf.load(conf_path)
    assert int(conf.outer_batch_size) > 1
    with JobContext(conf.run_name):
        run_rerank_with_conf_common(conf, get_scorer_fn)


if __name__ == "__main__":
    main()
