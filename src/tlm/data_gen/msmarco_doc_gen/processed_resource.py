import abc
from abc import ABC
from typing import List, Iterable, Callable, Dict, Tuple, Set


from data_generator.tokenizer_wo_tf import get_tokenizer
from dataset_specific.msmarco.common import QueryID, load_query_group, load_candidate_doc_list_1, SimpleQrel, \
    load_msmarco_simple_qrels, load_queries, load_token_d_1, load_candidate_doc_list_10, load_token_d_10doc, \
    load_candidate_doc_top50, load_token_d_50doc, top100_doc_ids, load_token_d_title_body


class ProcessedResourceI(ABC):
    def __init__(self, split):
        pass

    @abc.abstractmethod
    def get_doc_tokens_d(self, qid: QueryID) -> Dict[str, List[str]]:
        pass

    @abc.abstractmethod
    def get_label(self, qid: QueryID, doc_id):
        pass

    @abc.abstractmethod
    def get_q_tokens(self, qid: QueryID):
        pass

    @abc.abstractmethod
    def get_doc_for_query_d(self):
        pass

    @abc.abstractmethod
    def query_in_qrel(self, query_id):
        pass


class ProcessedResource(ProcessedResourceI):
    def __init__(self, split):
        super(ProcessedResource, self).__init__(split)
        query_group: List[List[QueryID]] = load_query_group(split)
        candidate_docs_d: Dict[QueryID, List[str]] = load_candidate_doc_list_1(split)
        qrel: SimpleQrel = load_msmarco_simple_qrels(split)

        self.split = split
        self.queires = dict(load_queries(split))
        self.query_group = query_group
        self.tokenizer = get_tokenizer()
        self.candidate_doc_d: Dict[QueryID, List[str]] = candidate_docs_d
        self.qrel = qrel
        self.tokenizer = get_tokenizer()

    def get_doc_tokens_d(self, qid: QueryID) -> Dict[str, List[str]]:
        return load_token_d_1(self.split, qid)

    def get_label(self, qid: QueryID, doc_id):
        return self.qrel.get_label(qid, doc_id)

    def get_q_tokens(self, qid: QueryID):
        query_text = self.queires[qid]
        return self.tokenizer.tokenize(query_text)

    def query_in_qrel(self, query_id):
        return query_id in self.qrel.qrel_d

    def get_doc_for_query_d(self):
        return self.candidate_doc_d


class ProcessedResource10doc(ProcessedResourceI):
    def __init__(self, split):
        super(ProcessedResource10doc, self).__init__(split)
        query_group: List[List[QueryID]] = load_query_group(split)
        candidate_docs_d: Dict[QueryID, List[str]] = load_candidate_doc_list_10(split)
        qrel: SimpleQrel = load_msmarco_simple_qrels(split)

        self.split = split
        self.queires = dict(load_queries(split))
        self.query_group = query_group
        self.tokenizer = get_tokenizer()
        self.candidate_doc_d: Dict[QueryID, List[str]] = candidate_docs_d
        self.qrel = qrel
        self.tokenizer = get_tokenizer()

    def get_doc_tokens_d(self, qid: QueryID) -> Dict[str, List[str]]:
        return load_token_d_10doc(self.split, qid)

    def get_label(self, qid: QueryID, doc_id):
        return self.qrel.get_label(qid, doc_id)

    def get_q_tokens(self, qid: QueryID):
        query_text = self.queires[qid]
        return self.tokenizer.tokenize(query_text)

    def get_doc_for_query_d(self):
        return self.candidate_doc_d

    def query_in_qrel(self, query_id):
        return query_id in self.qrel.qrel_d


class ProcessedResource50doc(ProcessedResourceI):
    def __init__(self, split):
        super(ProcessedResource50doc, self).__init__(split)
        query_group: List[List[QueryID]] = load_query_group(split)
        candidate_docs_d: Dict[QueryID, List[str]] = load_candidate_doc_top50(split)
        qrel: SimpleQrel = load_msmarco_simple_qrels(split)

        self.split = split
        self.queires = dict(load_queries(split))
        self.query_group = query_group
        self.tokenizer = get_tokenizer()
        self.candidate_doc_d: Dict[QueryID, List[str]] = candidate_docs_d
        self.qrel = qrel
        self.tokenizer = get_tokenizer()

    def get_doc_tokens_d(self, qid: QueryID) -> Dict[str, List[str]]:
        return load_token_d_50doc(self.split, qid)

    def get_label(self, qid: QueryID, doc_id):
        return self.qrel.get_label(qid, doc_id)

    def get_q_tokens(self, qid: QueryID):
        query_text = self.queires[qid]
        return self.tokenizer.tokenize(query_text)

    def get_doc_for_query_d(self):
        return self.candidate_doc_d

    def query_in_qrel(self, query_id):
        return query_id in self.qrel.qrel_d


class ProcessedResourcePredict:
    def __init__(self, split):
        query_group: List[List[QueryID]] = load_query_group(split)
        candidate_docs_d: Dict[QueryID, List[str]] = top100_doc_ids(split)
        qrel: SimpleQrel = load_msmarco_simple_qrels(split)

        self.split = split
        self.queires = dict(load_queries(split))
        self.query_group = query_group
        self.tokenizer = get_tokenizer()
        self.candidate_doc_d: Dict[QueryID, List[str]] = candidate_docs_d
        self.qrel = qrel
        self.tokenizer = get_tokenizer()

    def get_doc_tokens_d(self, qid: QueryID) -> Dict[str, List[str]]:
        return load_token_d_1(self.split, qid)

    def get_label(self, qid: QueryID, doc_id):
        return self.qrel.get_label(qid, doc_id)

    def get_q_tokens(self, qid: QueryID):
        query_text = self.queires[qid]
        return self.tokenizer.tokenize(query_text)


class ProcessedResourceTitleBodyTrain:
    def __init__(self, split, load_candidate_doc_list_fn):
        query_group: List[List[QueryID]] = load_query_group(split)
        candidate_docs_d: Dict[QueryID, List[str]] = load_candidate_doc_list_fn(split)
        qrel: SimpleQrel = load_msmarco_simple_qrels(split)

        self.split = split
        self.queires = dict(load_queries(split))
        self.query_group = query_group
        self.tokenizer = get_tokenizer()
        self.candidate_doc_d: Dict[QueryID, List[str]] = candidate_docs_d
        self.qrel = qrel
        self.tokenizer = get_tokenizer()

    def get_doc_tokens_d(self, qid: QueryID) -> Dict[str, Tuple[List[str], List[str]]]:
        return load_token_d_title_body(self.split, qid)

    def get_label(self, qid: QueryID, doc_id):
        return self.qrel.get_label(qid, doc_id)

    def get_q_tokens(self, qid: QueryID):
        query_text = self.queires[qid]
        return self.tokenizer.tokenize(query_text)

    def get_doc_for_query_d(self):
        return self.candidate_doc_d

    def query_in_qrel(self, query_id):
        return query_id in self.qrel.qrel_d


class ProcessedResourceTitleBodyPredict:
    def __init__(self, split):
        query_group: List[List[QueryID]] = load_query_group(split)
        candidate_docs_d: Dict[QueryID, List[str]] = top100_doc_ids(split)
        qrel: SimpleQrel = load_msmarco_simple_qrels(split)

        self.split = split
        self.queires = dict(load_queries(split))
        self.query_group = query_group
        self.tokenizer = get_tokenizer()
        self.candidate_doc_d: Dict[QueryID, List[str]] = candidate_docs_d
        self.qrel = qrel
        self.tokenizer = get_tokenizer()

    def get_doc_tokens_d(self, qid: QueryID) -> Dict[str, Tuple[List[str], List[str]]]:
        return load_token_d_title_body(self.split, qid)

    def get_label(self, qid: QueryID, doc_id):
        return self.qrel.get_label(qid, doc_id)

    def get_q_tokens(self, qid: QueryID):
        query_text = self.queires[qid]
        return self.tokenizer.tokenize(query_text)