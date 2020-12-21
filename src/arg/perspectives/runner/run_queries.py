import sys
from functools import partial
from typing import List, Dict

from arg.perspectives.basic_analysis import load_train_data_point
from arg.perspectives.dp_query_routines import dp_to_qid, db_not_contains, datapoint_to_doc_query
from arg.perspectives.ranked_list_interface import DynRankedListInterface, make_doc_query
from datastore.interface import has_key, save, flush
from datastore.table_names import QueryResult
from galagos.interface import DocQuery, send_doc_queries
from galagos.query_runs_ids import Q_CONFIG_ID_BM25_10000
from galagos.types import SimpleRankedListEntry
from list_lib import foreach, lfilter, lmap


##

def work(st, ed):
    st = int(st)
    ed = int(ed)
    q_config_id = Q_CONFIG_ID_BM25_10000
    ci = DynRankedListInterface(make_doc_query, q_config_id)
    all_data_points = load_train_data_point()

    print("Running {}~{} of {}".format(st, ed, len(all_data_points)))
    num_request = 10000
    todo = all_data_points[st:ed]
    not_done = lfilter(partial(db_not_contains, q_config_id), todo)
    queries: List[DocQuery] = lmap(datapoint_to_doc_query, not_done)
    print("Executing {} queries".format(len(queries)))
    ranked_list_dict: Dict[str, List[SimpleRankedListEntry]] = \
        send_doc_queries(ci.disk_path, num_request, queries, 600)
    qid_list = lmap(dp_to_qid, not_done)

    print("{} of {} succeed".format(len(ranked_list_dict), len(queries)))

    def add_to_db(query_id: str):
        if query_id in ranked_list_dict:
            r = ranked_list_dict[query_id]
            q_res_id: str = "{}_{}".format(query_id, q_config_id)
            if not has_key(QueryResult, q_res_id):
                save(QueryResult, q_res_id, r)

    foreach(add_to_db, qid_list)
    flush()


if __name__ == "__main__":
    work(sys.argv[1], sys.argv[2])
