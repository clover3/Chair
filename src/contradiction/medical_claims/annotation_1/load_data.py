import os
from typing import Dict, Tuple, List

from contradiction.medical_claims.annotation_1.annotation_html_gen import load_corpus
from cpath import output_path


def get_csv_path(group_no):
    input_csv_path = os.path.join(output_path,
                                  "alamri_annotation1",
                                  "grouped_pairs", "{}.csv".format(group_no))
    return input_csv_path


num_review = 24


def load_alamri1_all() -> List[Tuple[int, List[Tuple[str, str]]]]:
    all_pairs = []
    for i in range(1, 1+num_review):
        pairs: List[Tuple[str, str]] = load_corpus(get_csv_path(i))
        all_pairs.append((i, pairs))
    return all_pairs


def get_pair_dict() -> Dict[Tuple[int, int], Tuple[str, str]]:
    d = {}
    for group_no, items in load_alamri1_all():
        for idx, pair in enumerate(items):
            d[group_no, idx] = pair

    return d


