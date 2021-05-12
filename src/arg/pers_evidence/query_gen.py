import os
from collections import Counter
from typing import List, Iterable, Dict

from arg.perspectives.load import PerspectiveCluster, enum_perspective_clusters, get_all_claim_d, get_perspective_dict, \
    get_pc_cluster_query_id
from cpath import output_path
from galagos.interface import DocQuery, counter_to_galago_query
from galagos.parse import save_queries_to_file
from galagos.tokenize_util import TokenizerForGalago
from list_lib import lmap
from models.classic.lm_util import average_counters, sum_counters


def main():
    pc_clusters: Iterable[PerspectiveCluster] = enum_perspective_clusters()
    tokenizer = TokenizerForGalago()

    def get_terms(text: str) -> Counter:
        terms = tokenizer.tokenize(text)
        return Counter(terms)

    # Query = [claim :: avg(perspective)]
    claim_text_d: Dict[int, str] = get_all_claim_d()
    perspective_text_d: Dict[int, str] = get_perspective_dict()

    def cluster_to_query(cluster: PerspectiveCluster) -> DocQuery:
        claim_text = claim_text_d[cluster.claim_id]
        perspective_text_list = list([perspective_text_d[pid] for pid in cluster.perspective_ids])
        query_id = get_pc_cluster_query_id(cluster)
        claim_tf: Counter = get_terms(claim_text)
        pers_tf: Counter = average_counters(lmap(get_terms, perspective_text_list))
        tf = sum_counters([claim_tf, pers_tf])
        query: DocQuery = counter_to_galago_query(query_id, tf)
        return query

    query_list: List[DocQuery] = lmap(cluster_to_query, pc_clusters)
    print(len(query_list))
    out_path = os.path.join(output_path, "perspective_query",
                            "pc_query_for_evidence.json")
    save_queries_to_file(query_list, out_path)


if __name__ == "__main__":
    main()
