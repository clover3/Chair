from typing import List, Tuple, Dict

from adhoc.bm25_retriever import RetrieverIF
from misc_lib import TimeEstimator, get_second


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