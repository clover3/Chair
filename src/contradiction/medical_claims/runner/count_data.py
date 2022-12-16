from contradiction.medical_claims.load_corpus import load_parsed, Review
from typing import List, Iterable, Callable, Dict, Tuple, Set



def main():
    review_list: List[Review] = load_parsed()

    print("{} review".format(len(review_list)))
    n_claim = 0
    for review in review_list:
        n_claim += len(review.claim_list)
    print("{} claims".format(n_claim))


if __name__ == "__main__":
    main()

