from arg.perspectives.basic_analysis import PerspectiveCandidate
from arg.perspectives.ranked_list_interface import make_passage_query, make_doc_query
from datastore.interface import has_key
from datastore.table_names import QueryResult
from galagos.interface import PassageQuery, DocQuery


def dp_to_qid(dp: PerspectiveCandidate) -> str:
    query_id = "{}_{}".format(dp.cid, dp.pid)
    return query_id


def db_not_contains(q_config_id: int, dp: PerspectiveCandidate):
    query_id: str = dp_to_qid(dp)
    q_res_id: str = "{}_{}".format(query_id, q_config_id)
    return not has_key(QueryResult, q_res_id)


def datapoint_to_passage_query(dp:PerspectiveCandidate) -> PassageQuery:
    return make_passage_query(dp.cid, dp.pid, dp.claim_text, dp.p_text)


def datapoint_to_doc_query(dp:PerspectiveCandidate) -> DocQuery:
    return make_doc_query(dp.cid, dp.pid, dp.claim_text, dp.p_text)
