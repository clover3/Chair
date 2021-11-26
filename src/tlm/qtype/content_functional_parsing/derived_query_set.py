import os
from typing import List, Dict, Any

from cache import load_pickle_from
from cpath import qtype_root_dir
from dataset_specific.msmarco.common import QueryID
from misc_lib import group_by
from tlm.qtype.content_functional_parsing.qid_to_content_tokens import QueryInfo


class DerivedQuerySet:
    def __init__(self, raw_query_set: List[QueryInfo], query_mapping_fn):
        self.raw_query_set = raw_query_set
        self.qids_grouped: Dict[QueryID, List[QueryID]] = \
            group_by([q.qid for q in raw_query_set], query_mapping_fn)
        self.query_mapping_fn = query_mapping_fn
        self.query_info: Dict[QueryID, QueryInfo] = {e.qid: e for e in raw_query_set}

    def get_query(self, qid):
        return self.query_info[qid].query

    def get_new_query_grouped(self, query_group):
        new_q_group = []
        for q_group_entries in query_group:
            new_q_group_entries = []
            for qid in q_group_entries:
                new_q_group_entries.extend(self.qids_grouped[qid])
            new_q_group.append(new_q_group_entries)
        return new_q_group

    def extend_query_id_based_dict(self, d: Dict[QueryID, Any]):
        new_d = {}
        for qid, value in d.items():
            for new_qid in self.qids_grouped[qid]:
                new_d[new_qid] = value
        return new_d


def load_query_set_a(split) -> List[QueryInfo]:
    save_name = "NewQuerySetA{}".format(split)
    return load_pickle_from(os.path.join(qtype_root_dir, save_name))


def query_mapping_fn(query) -> QueryID:
    head, tail = query.split("_")
    return QueryID(head)


def load_derived_query_set_a(split) -> DerivedQuerySet:
    query_info_list = load_query_set_a(split)
    return DerivedQuerySet(query_info_list, query_mapping_fn)


def load_derived_query_set_a_small(split) -> DerivedQuerySet:
    query_info_list = load_query_set_a(split)
    seen = set()
    small_query_info_list = []
    for q in query_info_list:
        orig_qid = query_mapping_fn(q.qid)
        if orig_qid not in seen:
            seen.add(orig_qid)
            small_query_info_list.append(q)
    return DerivedQuerySet(small_query_info_list, query_mapping_fn)