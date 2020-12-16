from typing import List, Dict

from arg.perspectives.evaluate import perspective_getter
from arg.perspectives.load import load_claim_ids_for_split, enum_perspective_clusters_for_split, get_all_claim_d, \
    get_pc_cluster_query_id, enum_perspective_clusters, splits
from arg.qck.decl import QCKQuery, QKUnit
from cache import load_from_pickle


def get_qck_queries_all() -> List[QCKQuery]:
    pc_itr = enum_perspective_clusters()
    claim_text_d: Dict[int, str] = get_all_claim_d()

    query_list = []
    for pc in pc_itr:
        c_text = claim_text_d[pc.claim_id]
        pid = min(pc.perspective_ids)
        p_text = perspective_getter(pid)
        text = c_text + " " + p_text
        query = QCKQuery(get_pc_cluster_query_id(pc), text)
        query_list.append(query)

    return query_list


def get_qck_queries(split) -> List[QCKQuery]:
    claim_ids = set(load_claim_ids_for_split(split))
    pc_itr = enum_perspective_clusters_for_split(split)
    claim_text_d: Dict[int, str] = get_all_claim_d()

    query_list = []
    for pc in pc_itr:
        if pc.claim_id in claim_ids:
            c_text = claim_text_d[pc.claim_id]
            pid = min(pc.perspective_ids)
            p_text = perspective_getter(pid)
            text = c_text + " " + p_text
            query = QCKQuery(get_pc_cluster_query_id(pc), text)
            query_list.append(query)

    return query_list


def get_qk_per_split(source_pickle_name) -> Dict[str, List[QKUnit]]:
    qk_candidate: List[QKUnit] = load_from_pickle(source_pickle_name)
    qid_to_split = {}
    for split in splits:
        queries = get_qck_queries(split)
        for q in queries:
            qid_to_split[q.query_id] = split

    qk_per_split: Dict[str, List[QKUnit]] = {split: [] for split in splits}

    for q, k in qk_candidate:
        split = qid_to_split[q.query_id]
        qk_per_split[split].append((q, k))

    return qk_per_split
