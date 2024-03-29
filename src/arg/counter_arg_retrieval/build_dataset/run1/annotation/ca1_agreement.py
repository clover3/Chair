import sys
from collections import Counter
from typing import List, Any

from scipy.stats import pearsonr

from arg.counter_arg_retrieval.build_dataset.run1.annotation.scheme import get_ca_run1_scheme
from arg.counter_arg_retrieval.build_dataset.verify_common import print_hit_answers, kendalltau_fixed
from evals.agreement import cohens_kappa
from list_lib import lmap
from misc_lib import group_by, get_first, get_second, get_third, average
from mturk.parse_util import parse_file, HitResult


def summarize_agreement(hit_results: List[HitResult]):
    list_answers: List[List] = get_as_list_list(hit_results)

    annot1: List[Any] = lmap(get_first, list_answers)
    annot2: List[Any] = lmap(get_second, list_answers)
    try:
        annot3: List[Any] = lmap(get_third, list_answers)
    except IndexError:
        pass

    print("kappa")
    print('1 vs 2', kendalltau_fixed(annot1, annot2))
    print('2 vs 3', kendalltau_fixed(annot2, annot3))
    print('3 vs 1', kendalltau_fixed(annot3, annot1))

    def merge0_1(label):
        return {
            0: 0,
            1: 0,
            2: 1
        }[label]
    print("kappa ", cohens_kappa(annot1, annot2))
    annot1 = lmap(merge0_1, annot1)
    annot2 = lmap(merge0_1, annot2)
    print("kappa 0,1 vs 2", cohens_kappa(annot1, annot2))




def if_we_have_many_0(hit_results: List[HitResult]):
    list_answers: List[List] = get_as_list_list(hit_results)
    k = 100
    trivial = [0] * k

    annot1: List[Any] = lmap(get_first, list_answers)
    annot2: List[Any] = lmap(get_second, list_answers)

    print("before")
    print('kendal tau', kendalltau_fixed(annot1, annot2))
    print('cohen kappa', cohens_kappa(annot1, annot2))

    print("add {} 0's to {} data".format(k, len(annot1)))
    annot1 = annot1 + trivial
    annot2 = annot2 + trivial
    print("after")
    print('kendal tau', kendalltau_fixed(annot1, annot2))
    print('cohen kappa', cohens_kappa(annot1, annot2))



def pearson(hit_results: List[HitResult]):
    list_answers: List[List] = get_as_list_list(hit_results)

    annot1: List[Any] = lmap(get_first, list_answers)
    annot2: List[Any] = lmap(get_second, list_answers)
    annot3: List[Any] = lmap(get_third, list_answers)
    print("pearsonr")
    print('1 vs 2', pearsonr(annot1, annot2))
    print('2 vs 3', pearsonr(annot2, annot3))
    print('3 vs 1', pearsonr(annot3, annot1))


def get_as_list_list(hit_results):
    input_columns = list(hit_results[0].inputs.keys())
    answer_column = list(hit_results[0].outputs.keys())[0]

    def get_input_as_str(hit_result: HitResult):
        return "_".join([hit_result.inputs[key] for key in input_columns])

    list_answers = []
    for key, entries in group_by(hit_results, get_input_as_str).items():
        values = list([e.outputs[answer_column] for e in entries])
        list_answers.append(values)
    return list_answers


def show_agreement():
    hit_scheme = get_ca_run1_scheme()
    hit_results: List[HitResult] = parse_file(sys.argv[1], hit_scheme)
    summarize_agreement(hit_results)
    # pearson(hit_results)


def show_answer():
    hit_scheme = get_ca_run1_scheme()
    hit_results: List[HitResult] = parse_file(sys.argv[1], hit_scheme)
    print_hit_answers(hit_results)
    # pearson(hit_results)


def check_if_same_guy_always_wrong():
    path_file_to_validate = sys.argv[1]
    hit_scheme = get_ca_run1_scheme()
    hit_results = parse_file(path_file_to_validate, hit_scheme)
    input_columns = list(hit_results[0].inputs.keys())
    answer_column = list(hit_results[0].outputs.keys())[0]

    def get_input_as_str(hit_result: HitResult):
        return "_".join([hit_result.inputs[key] for key in input_columns])

    wrong_guy_count = Counter()
    guy_count = Counter()
    n_group = 0
    n_all_diff = 0
    for key, entries in group_by(hit_results, get_input_as_str).items():
        n_group += 1
        counter = Counter()
        for e in entries:
            c = e.outputs[answer_column]
            counter[c] += 1
            guy_count[e.worker_id] += 1

        common_answer, num_answer = list(counter.most_common(1))[0]
        if num_answer >= 2:
            for e in entries:
                c = e.outputs[answer_column]
                if c != common_answer:
                    wrong_guy_count[e.worker_id] += 1
        else:
            n_all_diff += 1
    print(n_all_diff, n_group)
    print(wrong_guy_count)

    for worker_id in wrong_guy_count:
        work_times = list([float(e.work_time) for e in hit_results if e.worker_id == worker_id])
        print(work_times)
        print("{}/{} : {} {}".format(wrong_guy_count[worker_id], guy_count[worker_id], worker_id, average(work_times)))


def test_if_we_have_many_0():
    hit_scheme = get_ca_run1_scheme()
    hit_results: List[HitResult] = parse_file(sys.argv[1], hit_scheme)
    if_we_have_many_0(hit_results)


def main():
    # check_if_same_guy_always_wrong()
    # test_if_we_have_many_0()
    # show_answer()
    show_agreement()


if __name__ == "__main__":
    main()