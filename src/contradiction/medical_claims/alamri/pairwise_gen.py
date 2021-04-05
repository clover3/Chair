from itertools import combinations
from typing import List, Iterable, Dict, Tuple

from contradiction.medical_claims.load_corpus import load_parsed, Review, Claim
from list_lib import lfilter


# 1. Contradiction:
#    - Yes/No claim from a same review
#    - No/Yes claim from a same review
# 2. Entail/Neutral:
#    - Yes/Yes or No/No claim from a same review
#    - Yes/No or No/Yes from two different reviews
#    - Yes/Yes or No/No claim from a different review
# 3.

def enum_true_instance() -> Iterable[Tuple[Claim, Claim, str]]:
    reviews: List[Review] = load_parsed()
    for review in reviews:
        yes_claim_list = lfilter(lambda c: c.assertion == "YS", review.claim_list)
        no_claim_list = lfilter(lambda c: c.assertion == "NO", review.claim_list)

        for yes_claim in yes_claim_list:
            for no_claim in no_claim_list:
                yield yes_claim, no_claim, "Yes/No from a same review"
                yield no_claim, yes_claim, "No/Yes from a same review"


def claim_text_to_info() -> Dict[str, Dict]:
    out_d = {}
    reviews: List[Review] = load_parsed()
    for review in reviews:
        for claim in review.claim_list:
            out_d[claim.text] = {
                'assertion': claim.assertion,
                'question': claim.question,
                'pmid': claim.pmid
            }
    return out_d


def enum_neg_instance() -> Iterable[Tuple[Claim, Claim, str]]:
    reviews: List[Review] = load_parsed()
    for review in reviews:
        yes_claim_list = lfilter(lambda c: c.assertion == "YS", review.claim_list)
        no_claim_list = lfilter(lambda c: c.assertion == "NO", review.claim_list)

        for c1, c2 in combinations(yes_claim_list, 2):
            yield c1, c2, "{}/{} from a same review".format(c1.assertion, c2.assertion)

        for c1, c2 in combinations(no_claim_list, 2):
            yield c1, c2, "{}/{} from a same review".format(c1.assertion, c2.assertion)

    for r1, r2 in combinations(reviews, 2):
        for c1 in r1.claim_list:
            for c2 in r2.claim_list:
                yield c1, c2, "{}/{} from different reviews".format(c1.assertion, c2.assertion)


def enum_neg_instance2() -> Iterable[Tuple[Claim, Claim, str]]:
    reviews: List[Review] = load_parsed()
    for review in reviews:
        yes_claim_list = lfilter(lambda c: c.assertion == "YS", review.claim_list)
        no_claim_list = lfilter(lambda c: c.assertion == "NO", review.claim_list)

        for c1, c2 in combinations(yes_claim_list, 2):
            yield c1, c2, "{}/{} from a same review".format(c1.assertion, c2.assertion)

        for c1, c2 in combinations(no_claim_list, 2):
            yield c1, c2, "{}/{} from a same review".format(c1.assertion, c2.assertion)


def enum_neg_instance_diff_review() -> Iterable[Tuple[Claim, Claim, str]]:
    reviews: List[Review] = load_parsed()
    for r1, r2 in combinations(reviews, 2):
        for c1 in r1.claim_list:
            for c2 in r2.claim_list:
                yield c1, c2, "{}/{} from different reviews".format(c1.assertion, c2.assertion)
