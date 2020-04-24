from typing import List

from galagos.types import Query
from misc_lib import ceil_divide
from tlm.alt_emb.xml_query_to_json import load_xml_query


def get_query_split():
    xml_path = "/mnt/nfs/work3/youngwookim/code/Chair/data/CLEFeHealth2017IRtask/queries/queries2016.xml"
    queries: List[Query] = load_xml_query(xml_path)

    n_query = len(queries)

    n_split = 5
    split_size = ceil_divide(n_query, n_split)

    cut = (n_split-1) * split_size
    train_queries = queries[:cut]
    test_queries = queries[cut:]

    return train_queries, test_queries