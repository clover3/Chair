import sys

from omegaconf import OmegaConf

from adhoc.bm25_class import BM25
from adhoc.other.bm25_retriever_helper import get_tokenize_fn
from dataset_specific.msmarco.passage.doc_indexing.retriever import get_bm25_stats_from_conf
from trainer_v2.per_project.transparency.mmp.bm25t_3 import BM25T_3
from trainer_v2.per_project.transparency.mmp.bm25t_helper import load_align_scores
from trainer_v2.per_project.transparency.mmp.eval_helper.rerank_w_conf import run_rerank_with_conf_common
from trainer_v2.per_project.transparency.mmp.parallel_helper import parallel_run
from trainer_v2.per_project.transparency.mmp.retrieval_run.run_bm25t import load_table_from_conf, to_value_dict
from typing import List, Iterable, Callable, Dict, Tuple, Set


def get_bm25t_scorer_fn(conf):
    print(conf)
    value_mapping: Dict[str, Dict[str, float]] = load_align_scores(conf.table_path)

    bm25_conf = OmegaConf.load(conf.bm25conf_path)
    avdl, cdf, df, dl = get_bm25_stats_from_conf(bm25_conf, None)
    tokenize_fn = get_tokenize_fn(bm25_conf)

    bm25 = BM25(df, cdf, avdl, 0.1, 100, 1.4)
    bm25t = BM25T_3(value_mapping, bm25.core, tokenize_fn)

    def parallel_score_fn(qd_list):
        split_n = 20
        if split_n > 1:
            ret = parallel_run(qd_list, bm25t.score_batch, split_n)
        else:
            ret = bm25t.score_batch(qd_list)
        return ret

    return parallel_score_fn


def main():
    get_scorer_fn = get_bm25t_scorer_fn
    conf_path = sys.argv[1]
    conf = OmegaConf.load(conf_path)
    assert int(conf.outer_batch_size) > 1
    run_rerank_with_conf_common(conf, get_scorer_fn)


if __name__ == "__main__":
    main()
