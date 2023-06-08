import csv
import os
from typing import List, Tuple

from contradiction.medical_claims.load_corpus import Review, Claim
from contradiction.medical_claims.pilot.pilot_annotation import enum_true_instance
from cpath import at_output_dir
from list_lib import foreach
from misc_lib import exist_or_mkdir


def main():
    save_dir = at_output_dir("alamri_annotation1", "grouped_pairs")
    exist_or_mkdir(save_dir)

    summary = []
    grouped_claim_pairs: List[Tuple[Review, List[Tuple[Claim, Claim]]]] = enum_true_instance(20)
    for review_idx, (review, claim_pairs) in enumerate(grouped_claim_pairs):
        entries = []
        for claim1, claim2 in claim_pairs:
            entries.append((claim1.text, claim2.text))

        review_no = review_idx + 1
        pair_save_path = os.path.join(save_dir, "{}.csv".format(review_no))
        csv_writer = csv.writer(open(pair_save_path, "w", newline='', encoding="utf-8"))
        foreach(csv_writer.writerow, entries)
        summary.append((str(review_no), review.pmid, str(len(claim_pairs))))

    summary_save_path = os.path.join(save_dir, 'summary.csv')
    csv_writer = csv.writer(open(summary_save_path, "w", newline='', encoding="utf-8"))
    foreach(csv_writer.writerow, summary)


if __name__ == "__main__":
    main()
