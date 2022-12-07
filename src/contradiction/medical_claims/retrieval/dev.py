from abc import ABC, abstractmethod
from typing import List, Iterable, Callable, Dict, Tuple, Set

from contradiction.medical_claims.annotation_1.load_data import load_reviews_for_split
from contradiction.medical_claims.load_corpus import Review


class BioClaimRetrievalSystem(ABC):
    def score(self, question, claim) -> float:
        pass


Queries = List[Tuple[str, str]]
Docs = List[Tuple[str, str]]


def get_bioclaim_retrieval_corpus(split) -> Tuple[Queries, Docs]:
    review_list: List[Tuple[int, Review]] = load_reviews_for_split(split)
    queries = []
    claims = []

    claim_unique = set()
    for group_no, r in review_list:
        query = r.claim_list[0].question
        qid = str(group_no)
        queries.append((qid, query))
        for c in r.claim_list:
            doc_id = c.pmid
            assert doc_id not in claim_unique
            claim_unique.add(doc_id)
            claims.append((doc_id, c.text))

    return queries, claims


def solve_BioClaim(system: BioClaimRetrievalSystem, split):
    pass


def main():
    # TODO  Task 1. Given an RQ, retrieve all relevant documents
    #   Method 1. Term-matching based
    #   Method 2. Large-LM approach
    #   Method 3. NLI driven approach


    # TODO  Task 2. Given an RQ, build a contradictory pair
    #   How to check polarity?
    #      Append 'YES' or 'No' to the question and do NLI classification.
    #      Method 1. Large-LM approach
    #      Method 2. NLITS


    return NotImplemented


if __name__ == "__main__":
    main()