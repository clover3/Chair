import sys

from omegaconf import OmegaConf

from adhoc.bm25_class import BM25
from dataset_specific.msmarco.passage.load_term_stats import load_msmarco_passage_term_stat
from trainer_v2.per_project.transparency.mmp.bm25t import BM25T
from trainer_v2.per_project.transparency.mmp.eval_helper.rerank_w_conf import run_rerank_with_conf_common
from trainer_v2.per_project.transparency.mmp.retrieval_run.run_bm25t import load_table_from_conf, to_value_dict


def get_bm25t_scorer_fn(conf):
    mapping = load_table_from_conf(conf)
    value_mapping = to_value_dict(mapping, conf.mapping_val)

    cdf, df = load_msmarco_passage_term_stat()
    bm25 = BM25(df, cdf, 25, 0.1, 100, 1.4)
    bm25t = BM25T(value_mapping, bm25.core)
    score_fn = bm25t.score_batch
    return score_fn


def main():
    get_scorer_fn = get_bm25t_scorer_fn
    conf_path = sys.argv[1]
    conf = OmegaConf.load(conf_path)
    run_rerank_with_conf_common(conf, get_scorer_fn)



if __name__ == "__main__":
    main()
