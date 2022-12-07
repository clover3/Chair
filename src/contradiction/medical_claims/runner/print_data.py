from contradiction.medical_claims.load_corpus import load_parsed, Review
from typing import List, Iterable, Callable, Dict, Tuple, Set



def main():
    review_list: List[Review] = load_parsed()

    for review in review_list:
        print("RQ\t", review.claim_list[0].question)

        for claim in review.claim_list:
            print(claim.assertion, "\t", claim.text)
        print()


if __name__ == "__main__":
    main()