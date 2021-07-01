import os
import sys
from collections import Counter
from typing import List, Any

from scipy.stats import pearsonr, kendalltau

from list_lib import lmap
from misc_lib import group_by, get_first, get_second, get_third
from mturk.parse_util import HITScheme, ColumnName, Checkbox, parse_file, HitResult, RadioButtonGroup
from stats.agreement import cohens_kappa


def get_ca_run1_scheme2():
    inputs = [ColumnName("c_text"), ColumnName("p_text"), ColumnName("doc_id")]
    # answer_list = [
    #     "Q1.on",
    #     "Q14.0.Q14.0",
    #     "Q14.1.Q14.1",
    #     "Q14.2.Q14.2",
    #     "Q14.3.Q14.3",
    #     "Q2.on",
    #     "claim_arg_oppose.on",
    #     "claim_arg_support.on",
    #     "claim_info_oppose.on",
    #     "claim_info_support.on",
    #     "claim_mention.on",
    #     "related.on",
    #     "topic_arg_oppose.on",
    #     "topic_arg_support.on",
    #     "topic_info_oppose.on",
    #     "topic_info_support.on",
    #     "topic_mention.on",
    # ]
    answer_units = []
    for i in range(1, 14):
        answer_units.append(Checkbox("Q{}.on".format(i)))

    answer_units.append(RadioButtonGroup("Q14.", lmap(str, range(4)), True))
    hit_scheme = HITScheme(inputs, answer_units)
    return hit_scheme


def summarize_agreement(hit_results: List[HitResult]):
    input_columns = list(hit_results[0].inputs.keys())
    answer_columns = list(hit_results[0].outputs.keys())

    def get_input_as_str(hit_result: HitResult):
        return "_".join([hit_result.inputs[key] for key in input_columns])

    answer_list_d = {}
    answer_list_d['key'] = []
    for key, entries in group_by(hit_results, get_input_as_str).items():
        if len(entries) < 3:
            print("skip", key)
        else:
            for answer_column in answer_columns:
                if answer_column not in answer_list_d:
                    answer_list_d[answer_column] = []
                answer_list_d[answer_column].append(list([e.outputs[answer_column] for e in entries]))
            answer_list_d['key'].append(key)
    return answer_list_d


def show_agreement():
    hit_scheme = get_ca_run1_scheme2()
    hit_results: List[HitResult] = parse_file(sys.argv[1], hit_scheme)
    answer_list_d = summarize_agreement(hit_results)
    for answer_column, list_answers in answer_list_d.items():
        annot1: List[Any] = lmap(get_first, list_answers)
        annot2: List[Any] = lmap(get_second, list_answers)
        annot3: List[Any] = lmap(get_third, list_answers)
        print(answer_column)
        print('1 vs 2', cohens_kappa(annot1, annot2))
        print('2 vs 3', cohens_kappa(annot2, annot3))
        print('3 vs 1', cohens_kappa(annot3, annot1))


def pearsonr_fixed(annot1, annot2):
    a, b = pearsonr(annot1, annot2)
    return "{0:.2f} {1:.2f}".format(a, b)


def kendalltau_fixed(annot1, annot2):
    a, b = kendalltau(annot1, annot2)
    return "{0:.2f} {1:.2f}".format(a, b)



def measure_correlation(hit_results: List[HitResult]):
    metric_fn = pearsonr_fixed
    answer_list_d = summarize_agreement(hit_results)
    for answer_column, list_answers in answer_list_d.items():
        annot1: List[Any] = lmap(get_first, list_answers)
        annot2: List[Any] = lmap(get_second, list_answers)
        annot3: List[Any] = lmap(get_third, list_answers)
        print(answer_column)
        print('1 vs 2', metric_fn(annot1, annot2))
        print('2 vs 3', metric_fn(annot2, annot3))
        print('3 vs 1', metric_fn(annot3, annot1))


def answers():
    hit_scheme = get_ca_run1_scheme2()
    hit_results: List[HitResult] = parse_file(sys.argv[1], hit_scheme)
    answer_list_d = summarize_agreement(hit_results)
    print(answer_list_d['key'])
    for answer_column, list_answers in answer_list_d.items():
        print(answer_column)
        print(list_answers)


def answers_per_input():
    hit_scheme = get_ca_run1_scheme2()
    hit_results: List[HitResult] = parse_file(sys.argv[1], hit_scheme)
    answer_list_d = summarize_agreement(hit_results)

    n_answers = 9999
    for k in range(100):
        if k >= n_answers:
            break

        for answer_column, list_answers in answer_list_d.items():
            print(answer_column)
            print(list_answers[k])
            n_answers = len(list_answers)


def count_counter_argument():
    common_dir = "C:\\work\\Code\\Chair\\output\\ca_building\\run1\\mturk_output"
    file_path = os.path.join(common_dir,  "Batch_4490355_batch_results.csv")
    hit_scheme = get_ca_run1_scheme2()
    hit_results: List[HitResult] = parse_file(file_path, hit_scheme)
    input_columns = list(hit_results[0].inputs.keys())

    def get_hit_key(hit_result: HitResult):
        return tuple([hit_result.inputs[key] for key in input_columns])
    answer_columns = list(hit_results[0].outputs.keys())

    counter = Counter()
    for key, entries in group_by(hit_results, get_hit_key).items():
        answer_column = "Q13"

        def get_answers_for_column(answer_column):
            return list([e.outputs[answer_column] for e in entries])

        for c in answer_columns:
            answers = get_answers_for_column(c)
            if sum(answers) >= 2:
                counter[c] += 1

        if len(entries) >= 2:
            counter[c] += 1



def main():
    answers_per_input()


def do_measure_correlation():
    common_dir = "C:\\work\\Code\\Chair\\output\\ca_building\\run1\\mturk_output"
    file_path_2 = os.path.join(common_dir,  "Batch_4490355_batch_results.csv")
    hit_scheme = get_ca_run1_scheme2()
    for file_path in [file_path_2]:
        hit_results: List[HitResult] = parse_file(file_path, hit_scheme)
        measure_correlation(hit_results)


if __name__ == "__main__":
    show_agreement()
