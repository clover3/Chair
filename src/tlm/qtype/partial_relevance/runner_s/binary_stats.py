from collections import Counter

from tlm.qtype.partial_relevance.related_answer_data_path_helper import load_binary_related_eval_answer


def stats(dataset, method):
    answers = load_binary_related_eval_answer(dataset, method)

    counter = Counter()
    for a in answers:
        for row in a.score_table:
            counter.update(row)

    print(method, counter)


def show_stats():
    method_list = ["exact_match", "gradient", "random", "exact_match_noise0.1", "exact_match_noise0.5"]
    dataset_name = "dev_sent"

    for method in method_list:
        stats(dataset_name, method)