import functools
import sys
from typing import List, Callable

from contradiction.alignment.data_structure.eval_data_structure import RelatedEvalAnswer, RelatedBinaryAnswer
from tlm.qtype.partial_relevance.related_answer_data_path_helper import load_related_eval_answer, \
    get_related_binary_save_path, save_json_at


def convert_answer(convert_fn: Callable[[float], int], a: RelatedEvalAnswer) -> RelatedBinaryAnswer:
    def convert_row(row):
        return list(map(convert_fn, row))

    new_table = list(map(convert_row, a.contribution.table))
    return RelatedBinaryAnswer(a.problem_id, new_table)


def discretize_and_save(dataset_name, method):
    answers: List[RelatedEvalAnswer] = load_related_eval_answer(dataset_name, method)
    cutoff = 0.5

    def convert(score):
        if score >= cutoff:
            return 1
        else:
            return 0

    convert_answer_fn = functools.partial(convert_answer, convert)
    new_answers: List[RelatedBinaryAnswer] = list(map(convert_answer_fn, answers))
    save_path = get_related_binary_save_path(dataset_name, method)
    save_json_at(new_answers, save_path)


def main():
    dataset_name = sys.argv[1]
    method = sys.argv[2]
    discretize_and_save(dataset_name, method)


if __name__ == "__main__":
    main()
