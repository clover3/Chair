import os
from typing import Dict, Tuple, List

from contradiction.medical_claims.annotation_1.annotation_html_gen import load_corpus
from contradiction.medical_claims.load_corpus import Review, load_parsed
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


def get_dev_group_no():
    return list(range(1, 13))


def get_test_group_no():
    return list(range(13, num_review))


def load_reviews_dev_split():
    reviews: List[Review] = load_parsed()
    dev_group_no_list = get_dev_group_no()
    output = []
    for group_no, review in enumerate(reviews):
        if group_no in dev_group_no_list:
            output.append((group_no, review))
    return output


def load_dev_pairs() -> List[Tuple[int, List[Tuple[str, str]]]]:
    dev_group_no_list = get_dev_group_no()

    def is_dev(e):
        group_no, _ = e
        return group_no in dev_group_no_list

    return list(filter(is_dev, load_alamri1_all()))


def load_dev_sents() -> List[Tuple[int, List[str]]]:
    pairs_grouped: List[Tuple[int, List[Tuple[str, str]]]] = load_dev_pairs()
    output: List[Tuple[int, List[str]]] = []
    for group_no, pairs in pairs_grouped:
        sent_set = set()
        for sent1, sent2 in pairs:
            sent_set.add(sent1)
            sent_set.add(sent2)
        output.append((group_no, list(sent_set)))
    return output


#
#
#