import sys
from typing import List, Callable

from tlm.qtype.partial_relevance.related_answer_data_path_helper import load_related_eval_answer, \
    get_related_binary_save_path, save_json_at, load_binary_related_eval_answer


def convert_answer(convert_fn: Callable[[float], int], a: RelatedEvalAnswer) -> RelatedBinaryAnswer:
    def convert_row(row):
        return list(map(convert_fn, row))

    new_table = list(map(convert_row, a.contribution.table))
    return RelatedBinaryAnswer(a.problem_id, new_table)


def count(int_array: List[int]):
    n = 0
    for s in int_array:
        assert s == 1 or s == 0
        if s:
            n += 1
    return n


def discretize_and_save(dataset_name, method_base, method_target):
    answers_base: List[RelatedBinaryAnswer] = load_binary_related_eval_answer(dataset_name, method_base)
    answers_target: List[RelatedEvalAnswer] = load_related_eval_answer(dataset_name, method_target)
    new_rba_list: List[RelatedBinaryAnswer] = []
    for base_answer, target_answer in zip(answers_base, answers_target):
        assert base_answer.problem_id == target_answer.problem_id
        n_row = len(base_answer.score_table)
        new_table = []
        for row_idx in range(n_row):
            n_true = count(base_answer.score_table[row_idx])
            scores = target_answer.contribution.table[row_idx]
            args = np.argsort(scores)[::-1]
            new_array = [0 for _ in scores]
            for i in args[:n_true]:
                new_array[i] = 1
            new_table.append(new_array)

        rba = RelatedBinaryAnswer(target_answer.problem_id, new_table)
        new_rba_list.append(rba)

    save_path = get_related_binary_save_path(dataset_name, method_target + "_cut")
    save_json_at(new_rba_list, save_path)


def main():
    dataset_name = sys.argv[1]
    method_base = sys.argv[2]
    method_target = sys.argv[3]
    discretize_and_save(dataset_name, method_base, method_target)


if __name__ == "__main__":
    main()
