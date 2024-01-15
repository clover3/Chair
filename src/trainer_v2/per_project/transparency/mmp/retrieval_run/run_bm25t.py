import sys
from collections import defaultdict

from omegaconf import OmegaConf
from typing import List, Iterable, Callable, Dict, Tuple, Set

from adhoc.bm25_retriever import build_bm25_scoring_fn
from adhoc.other.bm25t_retriever import BM25T_Retriever
from dataset_specific.msmarco.passage.doc_indexing.index_path_helper import BM25IndexResource
from dataset_specific.msmarco.passage.doc_indexing.retriever import load_bm25_resources
from dataset_specific.msmarco.passage.path_helper import TREC_DL_2019
from dataset_specific.msmarco.passage.trec_dl import run_mmp_retrieval_eval_report
from trainer_v2.per_project.transparency.mmp.table_readers import load_binary_mapping_from_align_candidate, \
    load_binary_mapping_from_align_scores
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.path_helper2 import load_omega_config_with_dataclass


def get_bm25t_retriever_from_conf(conf):
    bm25_conf = load_omega_config_with_dataclass(conf.bm25conf_path, BM25IndexResource)
    avdl, cdf, df, dl, inv_index = load_bm25_resources(bm25_conf)
    scoring_fn = build_bm25_scoring_fn(cdf, avdl)
    mapping_val = 0.1
    table = load_table_from_conf(conf)
    return BM25T_Retriever(inv_index, df, dl, scoring_fn, table, mapping_val)


def to_value_dict(table: Dict[str, List[str]], value: float) -> Dict[str, Dict[str, float]]:
    output_table = defaultdict(dict)
    for key, term_list in table.items():
        output_table[key] = {t: value for t in term_list}
    return output_table


def load_table_from_conf(conf) -> Dict[str, List[str]]:
    if conf.table_type == "candidates":
        table = load_binary_mapping_from_align_candidate(conf.table_path)
    elif conf.table_type == "score":
        cut = conf.cut
        if cut is None:
            cut = 0
        table = load_binary_mapping_from_align_scores(
            conf.table_path, cut)
    else:
        raise ValueError()
    return table


def main():
    conf_path = sys.argv[1]
    conf = OmegaConf.load(conf_path)
    dataset = TREC_DL_2019
    method = conf.method
    retriever = get_bm25t_retriever_from_conf(conf)
    run_mmp_retrieval_eval_report(dataset, method, retriever)


if __name__ == "__main__":
    main()