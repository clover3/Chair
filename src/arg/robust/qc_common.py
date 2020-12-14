import os
import random
from typing import Dict, List, Iterable

from arg.qck.decl import QCKCandidateWToken, QCKQuery
from cpath import data_path
from data_generator.data_parser.robust import load_robust04_title_query
from data_generator.data_parser.robust2 import load_bm25_best, load_qrel
from data_generator.tokenizer_wo_tf import get_tokenizer
from galagos.types import GalagoDocRankEntry
from list_lib import dict_value_map
from misc_lib import enum_passage
from tlm.robust.load import load_robust_tokens_for_predict, load_robust_tokens_for_train


def load_candidate_head_as_doc(doc_len=400) -> Dict[str, List[QCKCandidateWToken]]:
    top_k = 100
    candidate_docs: Dict[str, List[GalagoDocRankEntry]] = load_bm25_best()
    print("Num queries : ", len(candidate_docs))
    print("Loading robust collection tokens...", end= "")
    data: Dict[str, List[str]] = load_robust_tokens_for_predict()
    print("Done")
    print("Total of {} docs".format(len(data)))

    def make_candidate(doc_id: str):
        tokens = data[doc_id]
        return QCKCandidateWToken(doc_id, "", tokens[:doc_len])

    def fetch_docs(ranked_list: List[GalagoDocRankEntry]) -> List[QCKCandidateWToken]:
        return list([make_candidate(e.doc_id) for e in ranked_list[:top_k]])

    return dict_value_map(fetch_docs, candidate_docs)


def load_candidate_all_passage(max_seq_length, max_passage_per_doc=10) -> Dict[str, List[QCKCandidateWToken]]:
    candidate_docs: Dict[str, List[GalagoDocRankEntry]] = load_bm25_best()

    def get_doc_id(l: List[GalagoDocRankEntry]):
        return list([e.doc_id for e in l])

    candidate_doc_ids: Dict[str, List[str]] = dict_value_map(get_doc_id, candidate_docs)
    token_data: Dict[str, List[str]] = load_robust_tokens_for_predict()
    return load_candidate_all_passage_inner(candidate_doc_ids, token_data, max_seq_length, max_passage_per_doc)


def load_candidate_all_passage_inner(candidate_doc_ids, token_data,
                                     max_seq_length,
                                     max_passage_per_doc,
                                     top_k=100):
    print("Num queries : ", len(candidate_doc_ids))
    print("Loading robust collection tokens...", end="")
    print("Done")
    print("Total of {} docs".format(len(token_data)))
    tokenizer = get_tokenizer()
    d = {}
    for query, doc_ids in candidate_doc_ids.items():
        query_tokens = tokenizer.tokenize(query)
        content_len = max_seq_length - 3 - len(query_tokens)

        def make_candidate(doc_id: str) -> Iterable[QCKCandidateWToken]:
            tokens = token_data[doc_id]
            for idx, passage_tokens in enumerate(enum_passage(tokens, content_len)):
                if idx >= max_passage_per_doc:
                    break
                doc_part_id = "{}_{}".format(doc_id, idx)
                yield QCKCandidateWToken(doc_part_id, "", passage_tokens)

        insts_per_query = []
        for doc_id in doc_ids[:top_k]:
            for inst in make_candidate(doc_id):
                insts_per_query.append(inst)

        d[query] = insts_per_query
    return d


def load_candidate_all_passage_from_qrel(max_seq_length, max_passage_per_doc=10) -> Dict[str, List[QCKCandidateWToken]]:
    qrel_path = os.path.join(data_path, "robust", "qrels.rob04.txt")
    judgement: Dict[str, Dict] = load_qrel(qrel_path)

    candidate_doc_ids = {}
    for query_id in judgement.keys():
        judge_entries = judgement[query_id]
        doc_ids = list(judge_entries.keys())
        candidate_doc_ids[query_id] = doc_ids

    token_data = load_robust_tokens_for_train()

    return load_candidate_all_passage_inner(candidate_doc_ids,
                                     token_data,
                                     max_seq_length,
                                     max_passage_per_doc,
                                     9999
                                     )


def get_candidate_all_passage_w_samping(max_seq_length=256,
                                        neg_k=1000) -> Dict[str, List[QCKCandidateWToken]]:
    qrel_path = os.path.join(data_path, "robust", "qrels.rob04.txt")
    galago_rank = load_bm25_best()
    tokens_d = load_robust_tokens_for_train()
    tokens_d.update(load_robust_tokens_for_predict(4))
    queries = load_robust04_title_query()
    tokenizer = get_tokenizer()
    judgement: Dict[str, Dict] = load_qrel(qrel_path)
    out_d : Dict[str, List[QCKCandidateWToken]] = {}
    for query_id in judgement.keys():
        if query_id not in judgement:
            continue
        query = queries[query_id]
        query_tokens = tokenizer.tokenize(query)

        judge_entries = judgement[query_id]
        doc_ids = set(judge_entries.keys())

        ranked_list = galago_rank[query_id]
        ranked_list = ranked_list[:neg_k]
        doc_ids.update([e.doc_id for e in ranked_list])

        candidate = []
        for doc_id in doc_ids:
            tokens = tokens_d[doc_id]
            for idx, passage in enumerate(enum_passage(tokens, max_seq_length)):
                if idx == 0:
                    include = True
                else:
                    include = random.random() < 0.1

                if include:
                    c = QCKCandidateWToken(doc_id, "", passage)
                    candidate.append(c)

        out_d[query_id] = candidate
    return out_d


def get_candidate_all_passage_w_samping_predict(max_seq_length=256) -> Dict[str, List[QCKCandidateWToken]]:
    qrel_path = os.path.join(data_path, "robust", "qrels.rob04.txt")
    galago_rank = load_bm25_best()
    tokens_d = load_robust_tokens_for_predict(4)
    queries = load_robust04_title_query()
    tokenizer = get_tokenizer()
    out_d : Dict[str, List[QCKCandidateWToken]] = {}
    for query_id in queries:
        query = queries[query_id]
        query_tokens = tokenizer.tokenize(query)

        ranked_list = galago_rank[query_id]
        ranked_list = ranked_list[:100]
        doc_ids =list([e.doc_id for e in ranked_list])

        candidate = []
        for doc_id in doc_ids:
            tokens = tokens_d[doc_id]
            for idx, passage in enumerate(enum_passage(tokens, max_seq_length)):
                if idx == 0:
                    include = True
                else:
                    include = random.random() < 0.1

                if include:
                    c = QCKCandidateWToken(doc_id, "", passage)
                    candidate.append(c)

        out_d[query_id] = candidate
    return out_d


def to_qck_queries(queries):
    qck_queries = []
    for qid, query_text in queries.items():
        e = QCKQuery(qid, query_text)
        qck_queries.append(e)
    return qck_queries