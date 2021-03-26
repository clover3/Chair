from typing import List

from contradiction.medical_claims.load_corpus import load_parsed, Review
from contradiction.medical_claims.load_pubmed_doc import load_doc_parsed, Abstract
from contradiction.medical_claims.trivial_solver import solve, pairwise


def main():
    reviews: List[Review] = load_parsed()

    review0: Review = reviews[0]

    is_first = True
    for idx, claim in enumerate(review0.claim_list):
        doc: Abstract = load_doc_parsed(claim.pmid)
        if is_first:
            print("Question (query):", claim.question)
            is_first = False

        print("PMID:", claim.pmid)
        print("Claim {}:".format(idx), claim.text)
        if claim.assertion == "YS":
            print("YES")
        else:
            print(claim.assertion)
        print("Full abstract:")
        for text in doc.text_list:
            print("[{}] {}".format(text.label, text.text))

        print()


def run_trivial_solution():
    reviews: List[Review] = load_parsed()

    review0: Review = reviews[0]

    def solve_review(review: Review):
        question = review.claim_list[0].question
        claim_texts = list([c.text for c in review.claim_list])
        print("Question:", question)
        solve(question, claim_texts)

    solve_review(review0)


def run_trivial_solution_pair():
    reviews: List[Review] = load_parsed()

    review0: Review = reviews[0]

    def solve_review(review: Review):
        question = review.claim_list[0].question
        claim_texts = list([c.text for c in review.claim_list])
        claim_pairs = [(claim_texts[3], claim_texts[1])]
        # for idx in range(len(claim_texts)-1):
        #     c1 = claim_texts[idx]
        #     c2 = claim_texts[idx+1]
        #     claim_pairs.append((c1, c2))
        pairwise(claim_pairs)

    solve_review(review0)


if __name__ == "__main__":
    main()
