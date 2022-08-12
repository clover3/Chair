import random
from typing import List

from alignment.data_structure import ContributionSummary
from alignment.data_structure.eval_data_structure import Alignment2D
from tlm.qtype.partial_relevance.related_answer_data_path_helper import load_related_eval_answer, \
    save_related_eval_answer


def draw(prob):
    return random.random() < prob


def apply_noise(dataset, method, factor):
    answers: List[Alignment2D] = load_related_eval_answer(dataset, method)

    def is_true(s):
        return s > 1e-8

    n_cell = 0
    n_true = 0
    for a in answers:
        for row in a.contribution.table:
            n_cell += len(row)
            n_true += sum([1 if s > 1e-8 else 0 for s in row])

    true_rate = n_true / n_cell
    true_to_neg_rate = 0.5 * factor
    neg_to_true_rate = 0.5 * true_rate * factor

    def apply_noise_to_score(s):
        if is_true(s):
            if draw(true_to_neg_rate):
                new_s = 0
            else:
                new_s = 1
        else:  # false
            if draw(neg_to_true_rate):
                new_s = 1
            else:
                new_s = 0
        return new_s

    def apply_noise_to_answer(a: Alignment2D) -> Alignment2D:
        new_table = []
        for row in a.contribution.table:
            new_row = list(map(apply_noise_to_score, row))
            new_table.append(new_row)

        c = ContributionSummary(new_table)
        new_answer = Alignment2D(a.problem_id, c)
        return new_answer
    new_answer_list = list(map(apply_noise_to_answer, answers))
    new_method_name = method + "_noise{}".format(factor)
    save_related_eval_answer(new_answer_list, dataset, new_method_name)


def main():
    factor = 0.1
    apply_noise("dev_sent", "exact_match", factor)


if __name__ == "__main__":
    main()