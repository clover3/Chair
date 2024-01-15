from typing import List, Tuple

from adhoc.retriever_if import RetrieverIF
from adhoc.json_run_eval_helper import save_json_qres
from adhoc.adhoc_retrieval import run_retrieval
from table_lib import tsv_iter
from trainer_v2.chair_logging import c_log


def load_queries_from_conf(conf) -> List[Tuple[str, str]]:
    itr = tsv_iter(conf.queries_path)
    return list(itr)


def run_retrieval_from_conf(conf, retriever: RetrieverIF):
    run_name = conf.run_name
    queries = load_queries_from_conf(conf)
    c_log.info("%d queries", len(queries))
    max_doc_per_query = 1000
    doc_score_d = run_retrieval(retriever, queries, max_doc_per_query)
    save_json_qres(run_name, doc_score_d)