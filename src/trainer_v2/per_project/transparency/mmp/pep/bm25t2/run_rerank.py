import logging
import sys
from omegaconf import OmegaConf
from typing import List, Iterable, Callable, Dict, Tuple, Set
from adhoc.bm25_retriever import build_bm25_scoring_fn
from dataset_specific.msmarco.passage.doc_indexing.index_path_helper import load_bm25_index_resource_conf
from dataset_specific.msmarco.passage.doc_indexing.retriever import get_bm25_stats_from_conf
from misc_lib import select_third_fourth
from table_lib import tsv_iter
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.definitions import ModelConfig256_1
from trainer_v2.per_project.transparency.mmp.pep.bm25t2.bm25t2_scorer import BM25T2
from trainer_v2.per_project.transparency.mmp.pep.demo_util import PEPLocalDecision
from trainer_v2.per_project.transparency.mmp.pep.local_decision_helper import load_ts_concat_local_decision_model
from trainer_v2.train_util.get_tpu_strategy import get_strategy


def get_scorer_fn(conf):
    model_config = ModelConfig256_1()
    model = load_ts_concat_local_decision_model(model_config, conf.model_path)
    pep = PEPLocalDecision(model_config, model_path=None, model=model)

    c_log.info("Building scorer")
    bm25_conf = load_bm25_index_resource_conf(conf.bm25conf_path)
    avdl, cdf, df, dl = get_bm25_stats_from_conf(bm25_conf)
    bm25_scoring_fn: Callable[[int, int, int, int], float] = build_bm25_scoring_fn(cdf, avdl)
    bm25t2 = BM25T2(pep.score_fn, bm25_scoring_fn, df)
    return bm25t2.score


def main():
    c_log.info(__file__)
    c_log.setLevel(logging.DEBUG)
    conf_path = sys.argv[1]
    conf = OmegaConf.load(conf_path)
    strategy = get_strategy()
    with strategy.scope():
        c_log.setLevel(logging.DEBUG)
        # run config
        # Dataset config
        dataset_conf_path = conf.dataset_conf_path
        dataset_conf = OmegaConf.load(dataset_conf_path)
        quad_tsv_path = dataset_conf.rerank_payload_path
        # Prediction
        score_fn = get_scorer_fn(conf)
        qd_iter: Iterable[Tuple[str, str]] = select_third_fourth(tsv_iter(quad_tsv_path))

        try:
            outer_batch_size = conf.outer_batch_size
        except AttributeError:
            outer_batch_size = 1

        seen_q = set()
        for q, d in qd_iter:
            if q in seen_q:
                continue
            arg = [(q, d)]
            seen_q.add(q)
            score_fn(arg)


if __name__ == "__main__":
    main()
