import json
import os
from collections import defaultdict
from typing import List, Dict, Tuple

from contradiction.medical_claims.annotation_1.label_processor import json_dict_list_to_annots
from contradiction.medical_claims.annotation_1.load_data import get_pair_dict, get_dev_group_no
from contradiction.medical_claims.label_structure import PairedIndicesLabel, AlamriLabelUnitT
from cpath import output_path
from misc_lib import group_by_first


def load_labels() -> Dict[Tuple[int, int], List[PairedIndicesLabel]]:
    d = defaultdict(list)
    for name in ["worker_J", "worker_Q"]:
        source_json_path = os.path.join(output_path, "alamri_annotation1",
                                        "label", name + ".json")
        maybe_list = json.load(open(source_json_path, "r"))
        labels: List[AlamriLabelUnitT] = json_dict_list_to_annots(maybe_list)
        for key, value in labels:
            d[key].append(value)
    return d


def enum_group_sentence_wise(group_no_list):
    text_data = get_pair_dict()
    label_d = load_labels()

    # load sentence in the dev split
    # print 'conflict' tags for each sentences
    for group_no in group_no_list:
        inner_no_list = [i_no for g_no, i_no in text_data.keys() if g_no == group_no]
        inner_no_list.sort()
        unique_sents = set()
        flatten_tagged_sents = []

        for inner_no in inner_no_list:
            data_no = group_no, inner_no
            text1, text2 = text_data[data_no]
            unique_sents.add(text1)
            unique_sents.add(text2)

            labels = label_d[data_no]

            for label in labels:
                e = text1, label.prem_conflict_indices, label.prem_mismatch_indices
                flatten_tagged_sents.append(e)
                e = text2, label.hypo_conflict_indices, label.hypo_mismatch_indices
                flatten_tagged_sents.append(e)
        grouped_tagged_sents: Dict[str, List[Tuple[List[int], List[int]]]]\
            = group_by_first(flatten_tagged_sents)
        yield group_no, grouped_tagged_sents


def sent_pairwise_print(group_no_list):
    text_data = get_pair_dict()
    label_d = load_labels()

    # load sentence in the dev split
    # print 'conflict' tags for each sentences
    for group_no in group_no_list:
        inner_no_list = [i_no for g_no, i_no in text_data.keys() if g_no == group_no]
        inner_no_list.sort()
        print("Group {}".format(group_no))

        for inner_no in inner_no_list:
            data_no = group_no, inner_no
            text1, text2 = text_data[data_no]
            labels = label_d[data_no]
            for label in labels:
                print("")
                e = text1, label.prem_conflict_indices, label.prem_mismatch_indices
                e = text2, label.hypo_conflict_indices, label.hypo_mismatch_indices
                display_tokens = add_marker_to_tokens(label.prem_conflict_indices, text1.split())
                print(" ".join(display_tokens))
                display_tokens = add_marker_to_tokens(label.hypo_conflict_indices, text2.split())
                print(" ".join(display_tokens))


def print_groupwise():
    itr = enum_group_sentence_wise(get_dev_group_no())

    for group_no, grouped_tagged_sents in itr:
        print("Group {}".format(group_no))
        for sent, indices in grouped_tagged_sents.items():
            if not indices:
                continue
            tokens = sent.split()
            print("")
            for conflict_indices, mismatch_indices in indices:
                display_tokens = add_marker_to_tokens(conflict_indices, tokens)
                print(" ".join(display_tokens))


def add_marker_to_tokens(conflict_indices, tokens):
    display_tokens = []
    for i, token in enumerate(tokens):
        if i in conflict_indices:
            display_token = "[{}]".format(token)
        else:
            display_token = token
        display_tokens.append(display_token)
    return display_tokens


def main():
    sent_pairwise_print(get_dev_group_no())


if __name__ == "__main__":
    main()
