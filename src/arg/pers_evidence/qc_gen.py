from typing import List, Dict

from arg.perspectives.evaluate import perspective_getter
from arg.perspectives.load import load_claim_ids_for_split, enum_perspective_clusters_for_split, get_all_claim_d, \
    get_pc_cluster_query_id
from arg.qck.decl import QCKQuery


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
            #text_list = [c_text] + lmap(perspective_getter, pc.perspective_ids)
            #text = " ".join(text_list)
            text = c_text + " " + p_text
            query = QCKQuery(get_pc_cluster_query_id(pc), text)
            query_list.append(query)

    return query_list


if __name__ == "__main__":
    get_qck_queries("train")