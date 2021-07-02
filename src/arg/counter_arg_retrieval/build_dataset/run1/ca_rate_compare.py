import sys
from typing import List

from arg.counter_arg_retrieval.build_dataset.run1.agreement_scheme2 import get_ca_run1_scheme2
from arg.counter_arg_retrieval.build_dataset.run1.ca1_agreement import get_ca_run1_scheme
from arg.counter_arg_retrieval.build_dataset.run1.ca3_agreement import get_ca3_scheme
from misc_lib import SuccessCounter
from mturk.parse_util import parse_file, HitResult


def get_ca_rate_common(input_path, hit_scheme, get_binary_answer_from_hit_result):
    hit_results: List[HitResult] = parse_file(input_path, hit_scheme)
    sc = SuccessCounter()
    for hit_result in hit_results:
        if get_binary_answer_from_hit_result(hit_result):
            sc.suc()
        else:
            sc.fail()
    return sc


def get_ca1_ca_rate(input_path) -> SuccessCounter:
    hit_scheme = get_ca_run1_scheme()

    def get_binary_answer_from_hit_result(hit_result):
        worker_answer = hit_result.outputs['relevant.label']
        return worker_answer == 2

    return get_ca_rate_common(input_path, hit_scheme, get_binary_answer_from_hit_result)


def get_ca2_ca_rate(input_path) -> SuccessCounter:
    hit_scheme = get_ca_run1_scheme2()

    def get_binary_answer_from_hit_result(hit_result):
        answer_column = "Q13.on"
        worker_answer = hit_result.outputs[answer_column]
        return worker_answer == 1

    return get_ca_rate_common(input_path, hit_scheme, get_binary_answer_from_hit_result)


def get_ca3_ca_rate(input_path) -> SuccessCounter:
    hit_scheme = get_ca3_scheme()

    def get_binary_answer_from_hit_result(hit_result):
        answer_column = "Q2."
        worker_answer = hit_result.outputs[answer_column]
        return worker_answer == 1

    return get_ca_rate_common(input_path, hit_scheme, get_binary_answer_from_hit_result)


def main():
    ca1_path = sys.argv[1]
    ca2_path = sys.argv[2]
    ca3_path = sys.argv[3]
    sc1 = get_ca1_ca_rate(ca1_path)
    sc2 = get_ca2_ca_rate(ca2_path)
    sc3 = get_ca3_ca_rate(ca3_path)

    print(sc1.get_suc_prob())
    print(sc2.get_suc_prob())
    print(sc3.get_suc_prob())


if __name__ == "__main__":
    main()