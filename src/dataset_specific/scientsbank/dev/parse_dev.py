from collections import Counter

from dataset_specific.scientsbank.parse_fns import load_scientsbank_split, sci_ents_test_split_list
from dataset_specific.scientsbank.pte_data_types import Unaddressed, Expressed
from misc_lib import print_dict_tab

from typing import List


def main():
    todo: List[List[str]] = [["train"], sci_ents_test_split_list]

    reltn_label = Counter()
    reltn_set = set()
    counter = Counter()

    for split in sci_ents_test_split_list:
        questions = load_scientsbank_split(split, True)
        for q in questions:
            counter["question"] += 1
            counter["studentAnswers"] += len(q.student_answers)

            valid_facet_ids = set()
            ra = q.reference_answer
            counter["ref_facets"] += len(ra.facets)
            facet_to_reltn = {}
            for facet in ra.facets:
                counter["facet_valid"] += 1
                valid_facet_ids.add(facet.id)
                facet_to_reltn[facet.id] = facet.reltn
            for sa in q.student_answers:
                counter["facetEntailments"] += len(sa.facet_entailments)
                for fe in sa.facet_entailments:
                    if fe.label in [Expressed, Unaddressed]:
                        # if fe.facetID in valid_facet_ids:
                        counter["facetEntailments_valid"] += 1
                        key = "facetEntailments_" + fe.label
                        counter[key] += 1
                        reltn = facet_to_reltn[fe.facet_id]
                        reltn_label[reltn, fe.label] += 1
                        reltn_set.add(reltn)

    print_dict_tab(counter)
#
#     print(reltn_set)
#     reltn_list = list(reltn_set)
#     def bias(reltn):
#         pos = reltn_label[reltn, Expressed]
#         neg = reltn_label[reltn, Unaddressed]
#         return pos-neg
#
#     reltn_list.sort(key=bias)
#     pn_acc = 0
#     for reltn in reltn_list:
#         pos = reltn_label[reltn, Expressed]
#         neg = reltn_label[reltn, Unaddressed]
#         pn_acc += pos-neg
#         print(reltn, pos, neg, pos-neg, pn_acc)
#
#
#
#
# Expressed/Unaddressed
# 11895/12849
# -
# Exclude
# 9974/10516
#
# SemEval 2013 - Task7
# 4945/10318


if __name__ == "__main__":
    main()