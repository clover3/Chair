import sys
from typing import List, Any

from arg.counter_arg_retrieval.build_dataset.verify_common import summarize_agreement, annotator_eval
from list_lib import lmap
from misc_lib import get_first, get_second, get_third
from mturk.parse_util import HITScheme, ColumnName, parse_file, HitResult, YesNoRadioButtonGroup
from stats.agreement import cohens_kappa


def get_ca3_scheme():
    inputs = [ColumnName("c_text"), ColumnName("p_text"), ColumnName("doc_id")]
    q1 = YesNoRadioButtonGroup("Q1.")
    q2 = YesNoRadioButtonGroup("Q2.")
    answer_units = [q1, q2]
    hit_scheme = HITScheme(inputs, answer_units)
    return hit_scheme


def show_agreement():
    hit_scheme = get_ca3_scheme()
    hit_results: List[HitResult] = parse_file(sys.argv[1], hit_scheme)
    answer_list_d = summarize_agreement(hit_results)
    for answer_column, list_answers in answer_list_d.items():
        if answer_column == 'key':
            continue
        annot1: List[Any] = lmap(get_first, list_answers)
        annot2: List[Any] = lmap(get_second, list_answers)
        annot3: List[Any] = lmap(get_third, list_answers)
        print("annot1", annot1)
        print("annot2", annot2)
        print("annot3", annot3)
        print(answer_column)
        print('1 vs 2', cohens_kappa(annot1, annot2))
        print('2 vs 3', cohens_kappa(annot2, annot3))
        print('3 vs 1', cohens_kappa(annot3, annot1))

    correct_counter, wrong_counter = annotator_eval(hit_results)

    for key in wrong_counter.keys():
        print(key, correct_counter[key], wrong_counter[key])


def main():
    show_agreement()


if __name__ == "__main__":
    main()