from typing import List, Iterable, Dict, Tuple

from arg.perspectives.clueweb_db import load_doc
from arg.perspectives.doc_relevance.common import load_doc_scores
from arg.perspectives.load import d_n_claims_per_split2
from arg.perspectives.qck.qck_common import get_qck_queries
from arg.perspectives.runner_qck.run_qk_candidate_gen import config2
from arg.qck.decl import QCKQuery, KDP, QKUnit, KnowledgeDocumentPart, KnowledgeDocument
from arg.qck.kd_candidate_gen import get_duplicate, \
    iterate_document_parts
from cache import save_to_pickle
from datastore.interface import preload_man
from datastore.table_names import TokenizedCluewebDoc
from exec_lib import run_func_with_config
from galagos.parse import load_galago_ranked_list
from galagos.types import SimpleRankedListEntry
from list_lib import left
from misc_lib import TimeEstimator, get_second, tprint


# msmarco_qk_predict
# python src/arg/perspectives/show_qk_score.py
# gen_qk_candidate_with_score.py
# python src/arg/perspectives/runner_qck/gen_qk_candidate_with_score.py data/run_config/qk_candidate_with_score_dev.json
# python src/arg/perspectives/runner_qck/gen_qk_candidate_with_score.py data/run_config/qk_candidate_with_score_train.json

def iterate_docs(doc_ids: List[str]) -> Iterable[KnowledgeDocument]:
    docs = []
    for doc_id in doc_ids:
        try:
            tokens = load_doc(doc_id)
            kd = KnowledgeDocument(doc_id, tokens)
            docs.append(kd)
        except KeyError:
            pass

    if len(docs) < len(doc_ids):
        print("Retrieved {} of {} docs".format(len(docs), len(doc_ids)))
    duplicate_doc_ids = get_duplicate(docs)
    unique_docs = [d for d in docs if d.doc_id not in duplicate_doc_ids]
    return unique_docs


def qk_candidate_gen(q_res_path: str,
                     doc_score_path,
                     split,
                     config) -> List[Tuple[QCKQuery, List[KDP]]]:
    queries: List[QCKQuery] = get_qck_queries(split)
    num_jobs = d_n_claims_per_split2[split]
    score_d = load_doc_scores(doc_score_path, num_jobs)

    tprint("loading ranked list")
    ranked_list: Dict[str, List[SimpleRankedListEntry]] = load_galago_ranked_list(q_res_path)
    query_ids = list(ranked_list.keys())
    query_ids.sort()
    print("num queries", len(query_ids))
    q_id_to_job_id = {q_id: job_id for job_id, q_id in enumerate(query_ids)}
    print("Pre loading docs")
    top_n = config['top_n']
    out_qk: List[Tuple[QCKQuery, List[KnowledgeDocumentPart]]] = []

    all_doc_parts = 0
    ticker = TimeEstimator(len(queries))
    for q in queries:
        job_id: int = q_id_to_job_id[q.query_id]
        entries: List = score_d[job_id]
        entries.sort(key=get_second, reverse=True)
        doc_ids = left(entries)
        doc_ids = doc_ids[:top_n]
        preload_man.preload(TokenizedCluewebDoc, doc_ids)
        docs = iterate_docs(doc_ids)
        doc_part_list: List[KDP] = iterate_document_parts(docs,
                                                          config['window_size'],
                                                          config['step_size'],
                                                          20)

        all_doc_parts += len(doc_part_list)
        out_qk.append((q, doc_part_list))
        ticker.tick()
    return out_qk


def gen_overlap(config):
    split = config['split']
    q_res_path = config['q_res_path']
    save_name = config['save_name']
    doc_score_path = config['doc_score_path']
    candidate: List[QKUnit] = qk_candidate_gen(q_res_path, doc_score_path, split, config2())
    save_to_pickle(candidate, save_name)


if __name__ == "__main__":
    run_func_with_config(gen_overlap)
