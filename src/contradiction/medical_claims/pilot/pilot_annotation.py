import csv
from typing import List, Tuple

from contradiction.medical_claims.load_corpus import load_parsed, Review, Claim
from cpath import at_output_dir
from list_lib import lfilter, foreach


def num_common_terms(text1, text2):
    def get_terms(text):
        return set(text.lower().split())
    term1 = get_terms(text1)
    term2 = get_terms(text2)
    return len(term1.intersection(term2))


def enum_true_instance(sel_per_review=0) -> List[Tuple[Review, List[Tuple[Claim, Claim]]]]:
    reviews: List[Review] = load_parsed()

    def rank_fn(e: Tuple[Claim, Claim]):
        claim1, claim2 = e
        return num_common_terms(claim1.text, claim2.text)

    output = []
    for review in reviews:
        pair_per_review = []
        yes_claim_list = lfilter(lambda c: c.assertion == "YS", review.claim_list)
        no_claim_list = lfilter(lambda c: c.assertion == "NO", review.claim_list)

        for yes_claim in yes_claim_list:
            for no_claim in no_claim_list:
                e = yes_claim, no_claim
                pair_per_review.append(e)

        pair_per_review.sort(key=rank_fn, reverse=True)

        if sel_per_review == 0:
            pairs = pair_per_review
        else:
            pairs = pair_per_review[:sel_per_review]

        output.append((review, pairs))
    return output


def main():
    save_path = at_output_dir("alamri_pilot", "pilot_pairs.csv")

    entries = []
    for claim1, claim2 in enum_true_instance(3):
        print("--")
        print("{}".format(claim1.text))
        print("{}".format(claim2.text))
        entries.append((claim1.text, claim2.text))

    csv_writer = csv.writer(open(save_path, "w", newline='', encoding="utf-8"))
    foreach(csv_writer.writerow, entries)



if __name__ == "__main__":
    main()
