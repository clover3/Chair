from typing import Callable, List, Dict

from arg.pf_common.ranked_list_interface import RankedListInterface
from cache import load_from_pickle
from clueweb.clueweb_galago_db import load_from_db_or_from_galago
from dataset_specific.clue_path import get_first_disk
from datastore.table_names import QueryResult
from galagos.interface import send_queries_passage, PassageQuery, format_passage_query, DocQuery, \
    format_query_bm25, send_doc_queries
from galagos.tokenize_util import clean_tokenize_str_to_tokens
from galagos.types import SimpleRankedListEntry, GalagoPassageRankEntry


def make_passage_query(claim_id, perspective_id, claim_text, p_text) -> PassageQuery:
    query_id = "{}_{}".format(claim_id, perspective_id)
    raw_query_str: str = claim_text + " " + p_text
    q_terms: List[str] = clean_tokenize_str_to_tokens(raw_query_str)

    query = format_passage_query(query_id, q_terms)
    return query


def make_doc_query(claim_id, perspective_id, claim_text, p_text) -> DocQuery:
    query_id = "{}_{}".format(claim_id, perspective_id)
    raw_query_str: str = claim_text + " " + p_text
    q_terms: List[str] = clean_tokenize_str_to_tokens(raw_query_str)

    query = format_query_bm25(query_id, q_terms)
    return query



class PassageRankedListInterface:
    def __init__(self,
                 make_query_fn: Callable[[str, str, str, str], PassageQuery],
                 q_config_id: int):
        print("RankedListInterface __init__")
        self.disk_path = get_first_disk()
        self.collection_tf = load_from_pickle("collection_tf")
        self.make_query_fn = make_query_fn
        self.num_request = 10
        self.q_config_id = q_config_id

    def query_passage(self,
                      claim_id: str,
                      perspective_id: str,
                      claim_text: str,
                      p_text: str) -> List[GalagoPassageRankEntry]:
        query_id = "{}_{}".format(claim_id, perspective_id)
        print("query", query_id)
        # check if query_id is in DB
        q_res_id = "{}_{}".format(query_id, self.q_config_id)

        def galago_fn() -> List[GalagoPassageRankEntry]:
            print("running galago batch-search")
            query = make_passage_query(claim_id, perspective_id, claim_text, p_text)
            r: Dict[str, List[GalagoPassageRankEntry]] = send_queries_passage(self.disk_path, self.num_request, [query])
            return r[query_id]

        try:
            ranked_list = load_from_db_or_from_galago(QueryResult, q_res_id, galago_fn)
            return ranked_list
        except KeyError:
            print(query_id)
            print(ranked_list)
            raise


class DynRankedListInterface:
    def __init__(self,
                 make_query_fn: Callable[[str, str, str, str], DocQuery],
                 q_config_id: int):
        print("RankedListInterface __init__")
        self.disk_path = get_first_disk()
        self.collection_tf = load_from_pickle("collection_tf")
        self.make_query_fn = make_query_fn
        self.num_request = 10
        self.q_config_id = q_config_id

    def query(self,
              claim_id: str,
              perspective_id: str,
              claim_text: str,
              p_text: str) -> List[SimpleRankedListEntry]:
        query_id = "{}_{}".format(claim_id, perspective_id)
        #print("query", query_id)
        # check if query_id is in DB
        q_res_id = "{}_{}".format(query_id, self.q_config_id)

        def galago_fn() -> List[SimpleRankedListEntry]:
            print("running galago batch-search")
            query = make_doc_query(claim_id, perspective_id, claim_text, p_text)
            r: Dict[str, List[SimpleRankedListEntry]] = send_doc_queries(self.disk_path, self.num_request, [query])
            return r[query_id]

        try:
            ranked_list = load_from_db_or_from_galago(QueryResult, q_res_id, galago_fn)
            return ranked_list
        except KeyError:
            print(query_id)
            print(ranked_list)
            raise


class StaticRankedListInterface(RankedListInterface):
    def __init__(self,
                 q_config_id: int):
        print("RankedListInterface __init__")
        super(StaticRankedListInterface, self).__init__()
        self.num_request = 10
        self.q_config_id = q_config_id

    def fetch(self,
              claim_id: str,
              perspective_id: str
              ) -> List[SimpleRankedListEntry]:
        query_id = "{}_{}".format(claim_id, perspective_id)
        q_res_id = "{}_{}".format(query_id, self.q_config_id)
        return self.fetch_from_q_res_id(q_res_id)




