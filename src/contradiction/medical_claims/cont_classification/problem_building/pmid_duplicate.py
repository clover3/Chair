from typing import List, Iterable, Callable, Dict, Tuple, Set

from contradiction.medical_claims.annotation_1.load_data import load_reviews_for_split
from contradiction.medical_claims.load_corpus import Review


def main():
    split = "test"
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
            if doc_id in claim_unique:
                print("{} already appeared".format(doc_id))
                if claim_unique[doc_id] == c.text:
                    print("Test equal")
                else:
                    print("Test not equal")

            claim_unique[doc_id] = c.text
            claims.append((doc_id, c.text))




if __name__ == "__main__":
    main()