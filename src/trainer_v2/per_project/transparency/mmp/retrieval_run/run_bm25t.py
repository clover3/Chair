import sys

from omegaconf import OmegaConf

from adhoc.bm25_retriever import build_bm25_scoring_fn
from adhoc.other.bm25t_retriever import BM25T_Retriever
from dataset_specific.msmarco.passage.doc_indexing.index_path_helper import load_bm25_config
from dataset_specific.msmarco.passage.doc_indexing.retriever import load_bm25_resources
from dataset_specific.msmarco.passage.path_helper import TREC_DL_2019
from dataset_specific.msmarco.passage.trec_dl import run_mmp_test_retrieval_eval_report
from trainer_v2.per_project.transparency.mmp.bm25t_helper import load_binary_mapping_from_align_scores, \
    load_binary_mapping_from_align_candidate


def get_bm25t_retriever_from_conf(conf):
    bm25_config = load_bm25_config(conf.bm25conf_path)
    avdl, cdf, df, dl, inv_index = load_bm25_resources(bm25_config)
    scoring_fn = build_bm25_scoring_fn(cdf, avdl)
    mapping_val = 0.1
    if conf.table_type == "candidates":
        table = load_binary_mapping_from_align_candidate(conf.table_path)
    else:
        cut = conf.cut
        if cut is None:
            cut = 0
        table = load_binary_mapping_from_align_scores(
            conf.table_path, cut)
    return BM25T_Retriever(inv_index, df, dl, scoring_fn, table, mapping_val)


def main():
    conf_path = sys.argv[1]
    conf = OmegaConf.load(conf_path)
    dataset = TREC_DL_2019
    method = conf.method
    retriever = get_bm25t_retriever_from_conf(conf)
    run_mmp_test_retrieval_eval_report(dataset, method, retriever)


if __name__ == "__main__":
    main()