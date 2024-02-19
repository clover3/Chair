import itertools

from pytrec_eval import RelevanceEvaluator
from typing import List, Iterable, Callable, Dict, Tuple, Set, Iterator

from adhoc.adhoc_retrieval import run_retrieval
from adhoc.bm25_retriever import build_bm25_scoring_fn
from adhoc.eval_helper.pytrec_helper import load_qrels_as_structure_from_any
from adhoc.other.bm25_retriever_helper import get_stopwords_from_conf, load_bm25_resources, get_tokenize_fn, \
    build_bm25_scoring_fn_from_conf
from adhoc.other.bm25t_retriever import BM25T_Retriever2
from adhoc.other.index_reader_wrap import IndexReaderPython
from list_lib import lfrange
from tab_print import print_table
from table_lib import tsv_iter
from cpath import output_path, yconfig_dir_path
from misc_lib import path_join, average
from omegaconf import OmegaConf

from trainer_v2.per_project.transparency.mmp.retrieval_run.helper import get_dataset_conf_path


def get_resource(bm25_conf):
    avdl, cdf, df, dl, inv_index = load_bm25_resources(bm25_conf, None)
    tokenize_fn = get_tokenize_fn(bm25_conf)
    def get_posting(term):
        try:
            return inv_index[term]
        except KeyError:
            return []

    index_reader = IndexReaderPython(get_posting, df, dl)
    return index_reader, avdl, cdf, tokenize_fn


def main():
    dataset_conf_path = get_dataset_conf_path("dev_C")
    bm25conf_path = path_join(yconfig_dir_path, "bm25_resource", "lucene_krovetz.yaml")
    bm25_conf = OmegaConf.load(bm25conf_path)
    index_reader, avdl, cdf, tokenize_fn = get_resource(bm25_conf)

    dataset_conf = OmegaConf.load(dataset_conf_path)
    queries_path = dataset_conf.queries_path
    queries = list(tsv_iter(queries_path))
    metric = dataset_conf.metric
    judgment_path = dataset_conf.judgment_path
    max_doc_per_query = dataset_conf.max_doc_per_query

    table = {}
    stopwords = set()
    qrels: Dict[str, Dict[str, int]] = load_qrels_as_structure_from_any(judgment_path)

    def do_retrieval(retriever):
        doc_score_d = run_retrieval(retriever, queries, max_doc_per_query)
        evaluator = RelevanceEvaluator(qrels, {metric})
        score_per_query = evaluator.evaluate(doc_score_d)
        per_query_scores = [score_per_query[qid][metric] for qid in score_per_query]
        score = average(per_query_scores)
        return score

    b_range = lfrange(0.3, 0.4, 0.5)
    k1_range = lfrange(0.4, 0.5, 0.6)
    k2_range = [100]
    itr = itertools.product(b_range, k1_range, k2_range)

    res_table = []
    for param_cand in itr:
        b, k1, k2 = param_cand
        scoring_fn = build_bm25_scoring_fn(cdf, avdl, b, k1, k2)
        retriever = BM25T_Retriever2(index_reader, scoring_fn, tokenize_fn, table, stopwords)
        score = do_retrieval(retriever)
        row = (score, param_cand)
        print(row)
        res_table.append(row)

    print_table(res_table)


if __name__ == "__main__":
    main()
