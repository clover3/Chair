import functools
import sys
from typing import List

from contradiction.alignment.data_structure.eval_data_structure import RelatedEvalAnswer, RelatedBinaryAnswer, \
    convert_answer
from contradiction.alignment.related.related_answer_data_path_helper import load_related_eval_answer, \
    get_related_binary_save_path, save_json_at


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
