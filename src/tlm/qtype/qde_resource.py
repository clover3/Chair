from typing import List, Dict, Tuple

from dataset_specific.msmarco.common import QueryID
from tlm.data_gen.msmarco_doc_gen.processed_resource import ProcessedResourceTitleBodyTokensListI, \
    ProcessedResourceTitleBodyI
from tlm.qtype.content_functional_parsing.derived_query_set import DerivedQuerySet


class QDEResource(ProcessedResourceTitleBodyTokensListI):
    def __init__(self,
                 resource_source,
                 query_set: DerivedQuerySet,
                 ):
        print("QDEResource")
        print("Loading inner resource")
        self.resource_inner = resource_source
        self.query_set = query_set
        print("Getting grouped query...")
        self.query_group: List[List] = query_set.get_new_query_grouped(self.resource_inner.query_group)
        assert self.query_group
        assert self.query_group[0]

        self.qid_mapping = query_set.query_mapping_fn
        print("Mapping docs...")
        self.candidate_doc_d: Dict[QueryID, List[str]] = None

    def get_doc_tokens_d(self, qid: QueryID) -> Dict[str, Tuple[List[str], List[List[str]]]]:
        orig_qid = self.qid_mapping(qid)
        return self.resource_inner.get_doc_tokens_d(orig_qid)

    def get_label(self, qid: QueryID, doc_id):
        return 0

    def get_q_tokens(self, qid: QueryID):
        return self.resource_inner.tokenizer.tokenize(self.query_set.get_query(qid))

    def get_doc_for_query_d(self):
        if self.candidate_doc_d is None:
            self.candidate_doc_d: Dict[QueryID, List[str]] = \
                self.query_set.extend_query_id_based_dict(self.resource_inner.get_doc_for_query_d())
        return self.candidate_doc_d

    def query_in_qrel(self, query_id):
        return False


class QDEResourceFlat(ProcessedResourceTitleBodyI):
    def __init__(self,
                 resource_source,
                 query_set: DerivedQuerySet,
                 ):
        print("QDEResource")
        print("Loading inner resource")
        self.resource_inner = resource_source
        self.query_set = query_set
        print("Getting grouped query...")
        self.query_group: List[List] = query_set.get_new_query_grouped(self.resource_inner.query_group)
        assert self.query_group
        assert self.query_group[0]

        self.qid_mapping = query_set.query_mapping_fn
        print("Mapping docs...")
        self.candidate_doc_d: Dict[QueryID, List[str]] = \
            query_set.extend_query_id_based_dict(self.resource_inner.candidate_doc_d)

    def get_doc_tokens_d(self, qid: QueryID) -> Dict[str, Tuple[List[str], List[str]]]:
        orig_qid = self.qid_mapping(qid)
        return self.resource_inner.get_doc_tokens_d(orig_qid)

    def get_label(self, qid: QueryID, doc_id):
        return 0

    def get_q_tokens(self, qid: QueryID):
        return self.resource_inner.tokenizer.tokenize(self.query_set.get_query(qid))

    def get_doc_for_query_d(self):
        return self.candidate_doc_d

    def query_in_qrel(self, query_id):
        return False

