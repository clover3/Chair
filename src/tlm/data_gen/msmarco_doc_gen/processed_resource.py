import abc
import random
from abc import ABC
from typing import List, Dict, Tuple

from data_generator.tokenizer_wo_tf import get_tokenizer
from dataset_specific.msmarco.common import QueryID, load_query_group, load_candidate_doc_list_1, SimpleQrel, \
    load_msmarco_simple_qrels, load_queries, load_token_d_1, load_candidate_doc_list_10, load_token_d_10doc, \
    load_candidate_doc_top50, load_token_d_50doc, top100_doc_ids, load_token_d_title_body, load_multiple_resource, \
    load_token_d_sent_level
from list_lib import lfilter


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

    @abc.abstractmethod
    def get_query_text(self, qid: QueryID):
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

    def get_query(self, qid: QueryID):
        query_text = self.queires[qid]
        return query_text

    def get_query_text(self, qid: QueryID):
        query_text = self.queires[qid]
        return query_text

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

    def get_query_text(self, qid: QueryID):
        query_text = self.queires[qid]
        return query_text

    def get_q_tokens(self, qid: QueryID):
        query_text = self.queires[qid]
        return self.tokenizer.tokenize(query_text)

    def get_doc_for_query_d(self):
        return self.candidate_doc_d

    def query_in_qrel(self, query_id):
        return query_id in self.qrel.qrel_d


class ProcessedResourceMultiInterface:
    def __init__(self, split):
        query_group: List[List[QueryID]] = load_query_group(split)
        qrel: SimpleQrel = load_msmarco_simple_qrels(split)

        self.split = split
        self.queires = dict(load_queries(split))
        self.query_group = query_group
        self.tokenizer = get_tokenizer()
        self.qrel = qrel

    def get_stemmed_tokens_d(self, qid: QueryID) -> Dict[str, List[str]]:
        return load_multiple_resource(self.split, "stemmed_tokens", qid)

    def get_bert_tokens_d(self, qid: QueryID) -> Dict[str, List[str]]:
        return load_multiple_resource(self.split, "bert_tokens", qid)

    def get_text_d(self, qid: QueryID) -> Dict[str, List[str]]:
        return load_multiple_resource(self.split, "test", qid)

    def get_label(self, qid: QueryID, doc_id):
        return self.qrel.get_label(qid, doc_id)

    def get_query_text(self, qid: QueryID):
        query_text = self.queires[qid]
        return query_text

    def get_doc_for_query_d(self):
        return self.candidate_doc_d

    def query_in_qrel(self, query_id):
        return query_id in self.qrel.qrel_d


class ProcessedResource10docMulti(ProcessedResourceMultiInterface):
    def __init__(self, split):
        super(ProcessedResource10docMulti, self).__init__(split)
        candidate_docs_d: Dict[QueryID, List[str]] = load_candidate_doc_list_10(split)
        self.candidate_doc_d: Dict[QueryID, List[str]] = candidate_docs_d

    def get_doc_for_query_d(self):
        return self.candidate_doc_d


class ProcessedResource100docMulti(ProcessedResourceMultiInterface):
    def __init__(self, split):
        super(ProcessedResource100docMulti, self).__init__(split)
        candidate_docs_d: Dict[QueryID, List[str]] = top100_doc_ids(split)
        self.candidate_doc_d: Dict[QueryID, List[str]] = candidate_docs_d

    def get_doc_for_query_d(self):
        return self.candidate_doc_d


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

    def get_query_text(self, qid: QueryID):
        query_text = self.queires[qid]
        return query_text

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

    def get_candidate_doc_d(self, qid):
        return self.candidate_doc_d[qid]

    def get_doc_for_query_d(self):
        return self.candidate_doc_d


class ProcessedResourcePredict10(ProcessedResourcePredict):
    def __init__(self, split):
        super(ProcessedResourcePredict10, self).__init__(split)

        candidate_docs_d: Dict[QueryID, List[str]] = top100_doc_ids(split)
        new_candidate_docs_d: Dict[QueryID, List[str]] = {}
        for qid, doc_ids in candidate_docs_d.items():
            pos_doc_ids = lfilter(lambda doc_id: self.get_label(qid, doc_id), doc_ids)
            neg_doc_ids = lfilter(lambda doc_id: not self.get_label(qid, doc_id), doc_ids)
            n_neg = 10 - len(pos_doc_ids)
            random.shuffle(neg_doc_ids)
            doc_ids_selected = pos_doc_ids + neg_doc_ids[:n_neg]
            assert len(doc_ids_selected) <= 10
            new_candidate_docs_d[qid] = doc_ids_selected
        self.candidate_doc_d = new_candidate_docs_d




class ProcessedResourceTitleBodyI(ABC):
    @abc.abstractmethod
    def get_doc_tokens_d(self, qid: QueryID) -> Dict[str, Tuple[List[str], List[str]]]:
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



class ProcessedResourceTitleBodyTrain(ProcessedResourceTitleBodyI):
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


class ProcessedResourceTitleBodyPredict(ProcessedResourceTitleBodyI):
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

    def get_doc_for_query_d(self):
        return self.candidate_doc_d

    def query_in_qrel(self, query_id):
        return query_id in self.qrel.qrel_d



class ProcessedResourceTitleBodyTokensListI(ABC):
    @abc.abstractmethod
    def get_doc_tokens_d(self, qid: QueryID) -> Dict[str, Tuple[List[str], List[List[str]]]]:
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


class ProcessedResourceTitleBodyTokensListTrain(ProcessedResourceTitleBodyTokensListI):
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

    def get_doc_tokens_d(self, qid: QueryID) -> Dict[str, Tuple[List[str], List[List[str]]]]:
        return load_token_d_sent_level(self.split, qid)

    def get_label(self, qid: QueryID, doc_id):
        return self.qrel.get_label(qid, doc_id)

    def get_q_tokens(self, qid: QueryID):
        query_text = self.queires[qid]
        return self.tokenizer.tokenize(query_text)

    def get_doc_for_query_d(self):
        return self.candidate_doc_d

    def query_in_qrel(self, query_id):
        return query_id in self.qrel.qrel_d


class ProcessedResourceTitleBodyTokensListPredict(ProcessedResourceTitleBodyTokensListI):
    def __init__(self, split):
        query_group: List[List[QueryID]] = load_query_group(split)
        candidate_docs_d: Dict[QueryID, List[str]] = top100_doc_ids(split)
        qrel: SimpleQrel = load_msmarco_simple_qrels(split)
        self.queires = dict(load_queries(split))
        self.qrel = qrel
        self.split = split
        self.candidate_doc_d: Dict[QueryID, List[str]] = candidate_docs_d
        self.tokenizer = get_tokenizer()
        self.query_group = query_group

    def get_doc_tokens_d(self, qid: QueryID) -> Dict[str, Tuple[List[str], List[List[str]]]]:
        return load_token_d_sent_level(self.split, qid)

    def get_label(self, qid: QueryID, doc_id):
        return self.qrel.get_label(qid, doc_id)

    def get_q_tokens(self, qid: QueryID):
        query_text = self.queires[qid]
        return self.tokenizer.tokenize(query_text)

    def get_doc_for_query_d(self):
        return self.candidate_doc_d

    def query_in_qrel(self, query_id):
        return query_id in self.qrel.qrel_d

