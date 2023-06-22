import csv
from typing import List, Iterable, Callable, Dict, Tuple, Set
from cpath import output_path
from misc_lib import path_join

from dataset_specific.scientsbank.parse_fns import sci_ents_test_split_list, load_scientsbank_split, get_split_spec
from dataset_specific.scientsbank.pte_data_types import Question


def save_tsv(output, save_path):
    tsv_writer = csv.writer(open(save_path, "w", newline=""), delimiter="\t")
    tsv_writer.writerows(output)


def enum_sa_ra(questions: List[Question]):
    for q in questions:
        for sa in q.student_answers:
            prem = sa.answer_text
            hypo = q.reference_answer.text
            yield prem, hypo

def main():
    split_list = ["train_sub"] + sci_ents_test_split_list
    for split_name in split_list:
        split = get_split_spec(split_name)
        questions: List[Question] = load_scientsbank_split(split)

        save_path = path_join(output_path, "pte_scientsbank", f"plain_{split.get_save_name()}.txt")
        out_table = []
        for prem, hypo in enum_sa_ra(questions):
            out_table.append([prem, hypo])
        save_tsv(out_table, save_path)


if __name__ == "__main__":
    main()