import sys
from typing import Dict

from omegaconf import OmegaConf

from adhoc.conf_helper import load_omega_config
from adhoc.other.bm25_retriever_helper import get_tokenize_fn
from adhoc.other.ql_retriever_helper import load_ql_stats, build_ql_scoring_fn_from_conf
from adhoc.resource.qlt_method_loader import get_ql_conf_path
from taskman_client.wrapper3 import JobContext
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.bm25t.ql_rerank import QLRerank
from trainer_v2.per_project.transparency.mmp.eval_helper.rerank_w_conf import run_rerank_with_conf_common
from trainer_v2.per_project.transparency.mmp.table_readers import load_align_scores
import sys

from omegaconf import OmegaConf
from adhoc.conf_helper import create_omega_config
from adhoc.resource.dataset_conf_helper import get_dataset_conf
from adhoc.resource.scorer_loader import get_rerank_scorer, RerankScorerWrap
from taskman_client.wrapper3 import JobContext
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.eval_helper.rerank_w_conf import run_rerank_with_u_conf2, \
    RerankDatasetConf, RerankRunConf

def get_ql_scorer(conf, param):
    value_mapping: Dict[str, Dict[str, float]] = load_align_scores(conf.table_path)
    ql_conf = load_omega_config(conf.ql_conf_path, set_project_root=True)
    tokenize_fn = get_tokenize_fn(ql_conf)
    ql_conf.mu = param

    _avdl, _cdf, bg_prob, dl = load_ql_stats(ql_conf)
    scoring_fn = build_ql_scoring_fn_from_conf(ql_conf)

    ql_rerank = QLRerank(value_mapping, scoring_fn, tokenize_fn, bg_prob)
    rerank_scorer = RerankScorerWrap(ql_rerank.score_batch, False)
    return rerank_scorer


def run_rerank(param):
    dataset = "dev_c"
    table_path = "none"
    method = "ql_mu{}".format(param)
    scorer_conf = OmegaConf.create(
        {
            "ql_conf_path": get_ql_conf_path(),
            "table_path": table_path,
            "table_type": "Score",
            "method": method,
            "run_name": method
        }
    )
    rerank_scorer = get_ql_scorer(scorer_conf, param)
    dataset_conf: RerankDatasetConf = get_dataset_conf(dataset)
    outer_batch_size = rerank_scorer.get_outer_batch_size()
    conf: RerankRunConf = create_omega_config(
        {
            "dataset_conf": dataset_conf,
            "method": method,
            "run_name": method,
            "outer_batch_size": outer_batch_size,
        },  RerankRunConf
    )
    run_rerank_with_u_conf2(rerank_scorer.score_fn, conf)



def main():
    c_log.info(__file__)
    for param in [10, 100, 500, 2000]:
        run_rerank(param)


if __name__ == "__main__":
    main()
