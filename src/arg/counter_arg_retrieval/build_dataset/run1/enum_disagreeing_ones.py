import os
from collections import defaultdict

from arg.counter_arg_retrieval.build_dataset.run1.annotation.scheme import get_ca4_scheme, get_ca3_scheme
from cpath import output_path
from list_lib import lmap
from misc_lib import group_by, Averager
from mturk.parse_util import parse_file, HitResult

common_dir = os.path.join(output_path, "ca_building", "run1", "mturk_output")


def get_input_key(hit_result: HitResult):
    keys = ["c_text", "p_text", "doc_id"]
    l = list([hit_result.inputs[key] for key in keys])
    return tuple(l)


def do_for_ca3():
    hit_results = load_ca3_results1()

    target_answer = "Q2."
    for key, entries in group_by(hit_results, get_input_key).items():
        cnt = 0
        answers = list([e.outputs[target_answer] for e in entries])
        answers2 = list([e.outputs["Q2"] for e in entries])
        l = list(key) + lmap(str, answers) + answers2
        print("\t".join(l))


def load_ca3_results1():
    hit_scheme = get_ca3_scheme()
    ca3_input = os.path.join(common_dir, "CA3", "Batch_4493523_batch_results.csv")
    hit_results = parse_file(ca3_input, hit_scheme)
    return hit_results


def load_ca3_master():
    hit_scheme = get_ca3_scheme()
    ca3_input = os.path.join(common_dir, "CA3", "Batch_4506547_batch_results.csv")
    hit_results = parse_file(ca3_input, hit_scheme)
    return hit_results


def ca3_4_compare():
    ca4_shuffled_remain_A_master_path = os.path.join(ca4_path, "shuffled_remain_A_master.csv")
    ca4_shuffled_remain_B_master_path = os.path.join(ca4_path, "shuffled_remain_B_master.csv")
    hit_scheme = get_ca4_scheme()
    results_ca3 = load_ca3_master()
    results_ca4 = parse_file(ca4_shuffled_remain_A_master_path, hit_scheme)
    results_ca4 += parse_file(ca4_shuffled_remain_B_master_path, hit_scheme)

    ca4_d = defaultdict(list)
    for h in results_ca4:
        key = get_input_key(h)
        ca4_d[key].append(h)

    cnt_diff = 0
    cnt_common = 0
    keys = []

    n3_avg = Averager()
    n4_avg = Averager()

    for key, ca3_entries in group_by(results_ca3, get_input_key).items():
        if key in ca4_d:
            cnt_common += 1
            ca3_answers = list([e.outputs["Q2."] for e in ca3_entries])
            ca4_answers = list([e.outputs["Q8.on"] for e in ca4_d[key]])
            n3 = sum(ca3_answers)
            n4 = sum(ca4_answers)
            n3_avg.append(n3)
            n4_avg.append(n4)
            if n3 != n4:
                cnt_diff += 1
                print(n3, n4)
                keys.append(key)
            elif n3 == 3 and 3 == n4:
                print("BINGO")

    print("n3:", n3_avg.get_average())
    print("n4:", n4_avg.get_average())

    print(cnt_diff, cnt_common)
    for key in keys:
        print(",".join(key))



ca4_path = os.path.join(output_path, "ca_building", "run1", "mturk_output", "CA4")
def do_for_ca4():
    hit_scheme = get_ca4_scheme()
    # ca4_shuffled_remain_A_master_path = os.path.join(ca4_path, "shuffled_remain_A_master.csv")
    ca4_shuffled_remain_A_master_path = os.path.join(ca4_path, "shuffled_remain_A.csv")
    # ca4_shuffled_remain_B_master_path = os.path.join(ca4_path, "shuffled_remain_B_master.csv")
    hit_results = parse_file(ca4_shuffled_remain_A_master_path, hit_scheme)
    # hit_results += parse_file(ca4_shuffled_remain_B_master_path, hit_scheme)
    target_answer = "Q8.on"
    for key, entries in group_by(hit_results, get_input_key).items():
        answers = list([e.outputs[target_answer] for e in entries])
        l = list(key) + lmap(str, answers)
        print("\t".join(l))


def main():
    ca3_4_compare()


if __name__ == "__main__":
    main()