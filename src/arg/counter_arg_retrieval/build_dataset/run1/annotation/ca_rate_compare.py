import os

from arg.counter_arg_retrieval.build_dataset.run1.annotation.scheme import get_ca_run1_scheme, get_ca4_scheme, \
    get_ca_rate_common, \
    get_ca3_scheme, get_ca_run1_scheme2
from cpath import output_path
from misc_lib import SuccessCounter


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



def get_ca4_ca_rate(input_path) -> SuccessCounter:
    hit_scheme = get_ca4_scheme()

    def get_binary_answer_from_hit_result(hit_result):
        answer_column = "Q8.on"
        worker_answer = hit_result.outputs[answer_column]
        return worker_answer == 1

    return get_ca_rate_common(input_path, hit_scheme, get_binary_answer_from_hit_result)


common_dir = os.path.join(output_path, "ca_building", "run1", "mturk_output")
ca1_shuffle = os.path.join(common_dir, "CA1", "CA-1_shuffled.csv")
ca2_shuffle = os.path.join(common_dir, "CA2", "shuffled.csv")
ca3_path = os.path.join(common_dir, "CA3", "Batch_4506547_batch_results.csv")
ca4_dir = os.path.join(output_path, "ca_building", "run1", "mturk_output", "CA4")

def ca123():
    ca1_path = ca1_shuffle
    ca2_path = ca2_shuffle
    sc1 = get_ca1_ca_rate(ca1_path)
    sc2 = get_ca2_ca_rate(ca2_path)
    sc3 = get_ca3_ca_rate(ca3_path)

    print(sc1.get_suc_prob())
    print(sc2.get_suc_prob())
    print(sc3.get_suc_prob())


def ca34():
    ca4_shuffled_remain_A_master_path = os.path.join(ca4_dir, "shuffled_remain_A_master.csv")
    sc3 = get_ca3_ca_rate(ca3_path)
    sc4 = get_ca4_ca_rate(ca4_shuffled_remain_A_master_path)
    print(sc3.get_suc_prob())
    print(sc4.get_suc_prob())


if __name__ == "__main__":
    ca34()