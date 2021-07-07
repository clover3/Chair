import csv
from collections import Counter
from typing import Iterable, Dict
from typing import List, Any

from scipy.stats import pearsonr, kendalltau

from list_lib import lmap
from misc_lib import get_first, get_second, get_third, average
from misc_lib import group_by
from mturk.parse_util import HitResult


def get_list_of_known_labels(path, inputs: Iterable[str], answers: Iterable[str]) -> Dict:
    f = open(path, "r", encoding="utf-8")
    out_d = {}
    row_name_to_loc = {}
    for idx, row in enumerate(csv.reader(f)):
        if idx == 0:
            for row_idx, item in enumerate(row):
                row_name_to_loc[item] = row_idx
        else:
            input_values = get_column_values(inputs, row, row_name_to_loc)
            output_values = get_column_values(answers, row, row_name_to_loc)
            out_d[tuple(input_values)] = output_values
    return out_d


def get_column_values(columns, row, row_name_to_loc):
    input_keys = []
    for key in columns:
        input_keys.append(row[row_name_to_loc[key]])
    return input_keys


def summarize_agreement(hit_results: List[HitResult], min_entries=3):
    input_columns = list(hit_results[0].inputs.keys())
    answer_columns = list(hit_results[0].outputs.keys())

    def get_input_as_str(hit_result: HitResult):
        return "_".join([hit_result.inputs[key] for key in input_columns])

    answer_list_d = {}
    for key, entries in group_by(hit_results, get_input_as_str).items():
        if len(entries) < min_entries:
            print("skip", key)
        else:
            for answer_column in answer_columns:
                if answer_column not in answer_list_d:
                    answer_list_d[answer_column] = []
                answer_list_d[answer_column].append(list([e.outputs[answer_column] for e in entries]))
    return answer_list_d

def get_agreement_rate_from_answer_list(measure_fn, list_answers):
    annot1: List[Any] = lmap(get_first, list_answers)
    annot2: List[Any] = lmap(get_second, list_answers)
    annot3: List[Any] = lmap(get_third, list_answers)
    k12 = measure_fn(annot1, annot2)
    k23 = measure_fn(annot2, annot3)
    k31 = measure_fn(annot3, annot1)
    avg_k = average([k12, k23, k31])
    return avg_k


def show_agreement_inner(hit_results, measure_fn, scheme_question_d, drop_single_annotator):
    if drop_single_annotator:
        worker_count = Counter([h.worker_id for h in hit_results])
        hit_results = list([h for h in hit_results if worker_count[h.worker_id] > 1])
    answer_list_d = summarize_agreement(hit_results)
    for answer_column, list_answers in answer_list_d.items():
        avg_k = get_agreement_rate_from_answer_list(measure_fn, list_answers)
        print("{0}\t{1:.2f}\t{2}".format(answer_column, avg_k, scheme_question_d[answer_column]))


def count_true_rate(list_answers):
    n_true = 0
    n_all = 0
    for answers in list_answers:
        for answer in answers:
            n_all += 1
            if answer:
                n_true += 1
    return n_true/ n_all


def show_agreement_inner_w_true_rate(hit_results, measure_fn, scheme_question_d, drop_single_annotator):
    if drop_single_annotator:
        worker_count = Counter([h.worker_id for h in hit_results])
        hit_results = list([h for h in hit_results if worker_count[h.worker_id] > 1])
    answer_list_d = summarize_agreement(hit_results)
    for answer_column, list_answers in answer_list_d.items():
        avg_k = get_agreement_rate_from_answer_list(measure_fn, list_answers)
        true_rate = count_true_rate(list_answers)
        print("{0}\t{1:.2f}\t{2:.2f}\t{3}".format(answer_column, avg_k, true_rate, scheme_question_d[answer_column]))


def show_agreement_inner_for_two(hit_results, measure_fn, scheme_question_d, drop_single_annotator):
    if drop_single_annotator:
        worker_count = Counter([h.worker_id for h in hit_results])
        hit_results = list([h for h in hit_results if worker_count[h.worker_id] > 1])
    answer_list_d = summarize_agreement(hit_results, 2)
    for answer_column, list_answers in answer_list_d.items():
        annot1: List[Any] = lmap(get_first, list_answers)
        annot2: List[Any] = lmap(get_second, list_answers)
        k12 = measure_fn(annot1, annot2)
        print("{0}\t{1:.2f}\t{2}".format(answer_column, k12, scheme_question_d[answer_column]))



def annotator_eval(hit_results: List[HitResult]):
    input_columns = list(hit_results[0].inputs.keys())
    answer_columns = list(hit_results[0].outputs.keys())

    def get_input_as_str(hit_result: HitResult):
        return "_".join([hit_result.inputs[key] for key in input_columns])

    wrong_counter = Counter()
    correct_counter = Counter()
    for key, entries in group_by(hit_results, get_input_as_str).items():
        for answer_column in answer_columns:
            for e1 in entries:
                cur_answer = e1.outputs[answer_column]
                for e2 in entries:
                    if e1.worker_id == e2.worker_id:
                        continue
                    other_answer = e2.outputs[answer_column]

                    if cur_answer != other_answer:
                        wrong_counter[e1.worker_id] += 1
                    else:
                        correct_counter[e1.worker_id] += 1

    return correct_counter, wrong_counter


def print_hit_answers(hit_results):
    answer_list_d = summarize_agreement(hit_results, 0)
    n_answers = 9999
    for k in range(100):
        if k >= n_answers:
            break

        for answer_column, list_answers in answer_list_d.items():
            print(answer_column)
            print(list_answers[k])
            n_answers = len(list_answers)


def pearsonr_fixed(annot1, annot2):
    a, b = pearsonr(annot1, annot2)
    return "{0:.2f} {1:.2f}".format(a, b)


def kendalltau_fixed(annot1, annot2):
    a, b = kendalltau(annot1, annot2)
    return "{0:.2f} {1:.2f}".format(a, b)


def kendalltau_wrap(annot1, annot2):
    a, b = kendalltau(annot1, annot2)
    return a


def count_all_true(hit_results, column):
    answer_list_d = summarize_agreement(hit_results)
    list_answers = answer_list_d[column]
    all_true_cnt = 0
    total = 0
    for answers in list_answers:
        if len(answers) == 3 and all(answers):
            all_true_cnt += 1
        total += 1

    return all_true_cnt, total
