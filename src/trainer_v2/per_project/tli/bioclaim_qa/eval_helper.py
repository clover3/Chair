from typing import List, Callable, Tuple

from contradiction.medical_claims.annotation_1.load_data import load_reviews_for_split
from contradiction.medical_claims.load_corpus import Review
from trainer_v2.per_project.tli.bioclaim_qa.path_helper import get_retrieval_save_path
from trainer.promise import PromiseKeeper
from trainer_v2.chair_logging import c_log
from trec.ranked_list_util import build_ranked_list
from trec.trec_parse import write_trec_ranked_list_entry
from trec.types import TrecRankedListEntry, TrecRelevanceJudgementEntry

Queries = List[Tuple[str, str]]
Docs = List[Tuple[str, str]]


# Build Claim ID

def get_bioclaim_retrieval_corpus(split) -> Tuple[Queries, Docs]:
    review_list: List[Tuple[int, Review]] = load_reviews_for_split(split)
    queries = []
    claims = []

    claim_unique = dict()
    for group_no, r in review_list:
        query = r.claim_list[0].question
        qid = str(group_no)
        queries.append((qid, query))
        for c in r.claim_list:
            doc_id = c.pmid
            do_add = True
            while doc_id in claim_unique:
                if claim_unique[doc_id] == c.text:
                    do_add = False
                    break
                doc_id = c.pmid + "_2"

            if do_add:
                claim_unique[doc_id] = c.text
                claims.append((doc_id, c.text))

    return queries, claims


def build_qrel(split) -> List[TrecRelevanceJudgementEntry]:
    review_list: List[Tuple[int, Review]] = load_reviews_for_split(split)
    out_e = []
    for group_no, r in review_list:
        qid = str(group_no)
        for c in r.claim_list:
            doc_id = c.pmid
            e = TrecRelevanceJudgementEntry(qid, doc_id, 1)
            out_e.append(e)
    return out_e

TextPairScorer = Callable[[str, str], float]
BatchTextPairScorer = Callable[[List[Tuple[str, str]]], List[float]]


def solve_bioclaim(scorer: TextPairScorer, split, run_name)\
        -> List[TrecRankedListEntry]:
    queries, claims = get_bioclaim_retrieval_corpus(split)
    rl_flat = []
    for qid, query in queries:
        scored_docs = []
        for doc_id, claim in claims:
            score = scorer(query, claim)
            scored_docs.append((doc_id, score))

        rl = build_ranked_list(qid, run_name, scored_docs)
        rl_flat.extend(rl)
    return rl_flat


def batch_solve_bioclaim(scorer: BatchTextPairScorer, split, run_name, mini_debug=False)\
        -> List[TrecRankedListEntry]:
    queries, docs = get_bioclaim_retrieval_corpus(split)
    if mini_debug:
        queries = queries[:3]
        docs = docs[:3]

    pk = PromiseKeeper(scorer)
    qid_doc_id_future_list = []
    for qid, query in queries:
        for doc_id, claim in docs:
            score_f = pk.get_future((query, claim))
            qid_doc_id_future_list.append((qid, doc_id, score_f))

    pk.do_duty(True)
    score_d = {}
    for qid, doc_id, f in qid_doc_id_future_list:
        score_d[qid, doc_id] = f.get()

    rl_flat = []
    for qid, _ in queries:
        scored_docs = []
        for doc_id, claim in docs:
            score = score_d[qid, doc_id]
            scored_docs.append((doc_id, score))

        rl = build_ranked_list(qid, run_name, scored_docs)
        rl_flat.extend(rl)
    return rl_flat


def solve_bio_claim_and_save(scorer: TextPairScorer, split, run_name):
    rl_flat = solve_bioclaim(scorer, split, run_name)
    write_trec_ranked_list_entry(rl_flat, get_retrieval_save_path(run_name))


