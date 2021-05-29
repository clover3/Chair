import csv
import os
from typing import List, Iterable, Callable, Dict, Tuple, Set
from contradiction.medical_claims.load_corpus import Review, Claim
from cpath import output_path, at_output_dir
from list_lib import foreach
from misc_lib import DataIDManager, exist_or_mkdir


def main():
    save_dir = at_output_dir("alamri_annotation1", "grouped_pairs")
    exist_or_mkdir(save_dir)

    summary = []
    grouped_claim_pairs: List[Tuple[Review, List[Claim, Claim]]] = NotImplemented
    for review_idx, (review, claim_pairs) in enumerate(grouped_claim_pairs):
        entries = []
        for claim1, claim2 in claim_pairs:
            entries.append((claim1.text, claim2.text))

        review_no = review_idx + 1
        save_path = os.path.join(save_dir, "{}.csv".format(review_no))
        csv_writer = csv.writer(open(save_path, "w", newline='', encoding="utf-8"))
        foreach(csv_writer.writerow, entries)
        summary.append((str(review_no), review.pmid, str(len(claim_pairs))))

    save_path = os.path.join(save_dir, 'sumamry.csv')
    csv_writer = csv.writer(open(save_path, "w", newline='', encoding="utf-8"))
    foreach(csv_writer.writerow, summary)


if __name__ == "__main__":
    main()
