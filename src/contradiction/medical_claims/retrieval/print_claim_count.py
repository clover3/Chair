from collections import Counter
from typing import List, Iterable, Callable, Dict, Tuple, Set

from contradiction.medical_claims.annotation_1.load_data import load_reviews_for_split
from contradiction.medical_claims.load_corpus import Review


def main():
    for split in ["dev", "test"]:
        review_list: List[Tuple[int, Review]] = load_reviews_for_split(split)

        for _, review in review_list:
            counter = Counter()
            for c in review.claim_list:
                counter[c.assertion] += 1
            print(counter)



if __name__ == "__main__":
    main()