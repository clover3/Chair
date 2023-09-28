import json
from typing import Dict, List, Tuple

from adhoc.bm25_retriever import RetrieverIF
from dataset_specific.beir_eval.path_helper import get_json_qres_save_path
from misc_lib import TimeEstimator, get_second
from typing import List, Iterable, Callable, Dict, Tuple, Set



def save_json_qres(run_name: str, output):
    json_qres_save_path = get_json_qres_save_path(run_name)
    json.dump(output, open(json_qres_save_path, "w"))


def load_json_qres(run_name: str):
    json_qres_save_path = get_json_qres_save_path(run_name)
    return json.load(open(json_qres_save_path, "r"))


def run_retrieval(
        retriever: RetrieverIF,
        queries: List[Tuple[str, str]],
        max_doc_per_query) -> Dict[str, Dict[str, float]]:
    ticker = TimeEstimator(len(queries))
    output: Dict[str, Dict[str, float]] = {}
    for qid, query_text in queries:
        res: List[Tuple[str, float]] = retriever.retrieve(query_text)
        res.sort(key=get_second, reverse=True)

        per_query_res = {}
        for doc_id, score in res[:max_doc_per_query]:
            per_query_res[doc_id] = score

        output[qid] = per_query_res
        ticker.tick()
    return output