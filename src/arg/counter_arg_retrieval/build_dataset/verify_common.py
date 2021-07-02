import csv
from collections import Counter
from typing import Iterable, Dict, List

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