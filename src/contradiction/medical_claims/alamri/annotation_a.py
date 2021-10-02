from typing import List, Iterable, Tuple

from contradiction.medical_claims.load_corpus import load_parsed, Review, Claim
from contradiction.medical_claims.pilot.pilot_annotation import num_common_terms
from list_lib import lfilter
from tab_print import tab_print


def enum_true_instance(sel_per_review=0) -> Iterable[Tuple[Claim, Claim]]:
    reviews: List[Review] = load_parsed()

    def rank_fn(e: Tuple[Claim, Claim]):
        claim1, claim2 = e
        return num_common_terms(claim1.text, claim2.text)

    n_claim_acc = 0
    for review in reviews:
        pair_per_review = []
        yes_claim_list = lfilter(lambda c: c.assertion == "YS", review.claim_list)
        no_claim_list = lfilter(lambda c: c.assertion == "NO", review.claim_list)
        n_yes, n_no = len(yes_claim_list), len(no_claim_list)

        n_claim_acc += len(yes_claim_list)
        n_claim_acc += len(no_claim_list)
        tab_print(n_yes * n_no, n_yes, n_no)
        for yes_claim in yes_claim_list:
            for no_claim in no_claim_list:
                e = yes_claim, no_claim
                pair_per_review.append(e)

        pair_per_review.sort(key=rank_fn, reverse=True)

        if sel_per_review == 0:
            pairs = pair_per_review
        else:
            pairs = pair_per_review[:sel_per_review]

        for claim1, claim2 in pairs:
            yield claim1, claim2

    print("Total of {} claims".format(n_claim_acc))


def main():
    pairs = list(enum_true_instance())
    print("total {} pairs".format(len(pairs)))


if __name__ == "__main__":
    main()