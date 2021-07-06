import os
import sys
from collections import Counter
from typing import List

from arg.counter_arg_retrieval.build_dataset.run1.ca_rate_compare import get_ca_rate_common
from arg.counter_arg_retrieval.build_dataset.verify_by_acess_log import load_apache_log, verify_by_time, verify_by_ip, \
    ApacheLogParsed
from arg.counter_arg_retrieval.build_dataset.verify_common import summarize_agreement, print_hit_answers, \
    show_agreement_inner_for_two
from cpath import output_path
from list_lib import lmap
from misc_lib import group_by, average, SuccessCounter
from mturk.parse_util import HITScheme, ColumnName, Checkbox, parse_file, HitResult, RadioButtonGroup
from stats.agreement import cohens_kappa

scheme4_question_d = {
    "Q1.on": "claim supports topic",
    "Q2.on": "claim opposes topic",
    "Q3.on": "document mentions topic",
    "Q4.on": "document mentions claim",
    "Q5.on": "document contains information supporting topic",
    "Q6.on": "document contains information opposing topic ",
    "Q7.on": "document contains information supporting claim",
    "Q8.on": "document contains information opposing claim",
    "Q9.": "useful?"
}

def get_ca4_scheme():
    inputs = [ColumnName("c_text"), ColumnName("p_text"), ColumnName("doc_id")]
    answer_units = []
    for i in range(1, 9):
        answer_units.append(Checkbox("Q{}.on".format(i)))
    answer_units.append(RadioButtonGroup("Q9.", lmap(str, range(4)), True))
    hit_scheme = HITScheme(inputs, answer_units)
    return hit_scheme


def answers_per_input():
    hit_scheme = get_ca4_scheme()
    hit_results: List[HitResult] = parse_file(sys.argv[1], hit_scheme)
    print_hit_answers(hit_results)


def verify_by_logs_dev():
    hit_scheme = get_ca4_scheme()
    hit_results: List[HitResult] = parse_file(sys.argv[1], hit_scheme)
    logs: List[ApacheLogParsed] = load_apache_log()
    suspicious_hits, ip_worker_matches = verify_by_time(hit_results, logs)
    print(len(suspicious_hits), len(hit_results))
    for h in suspicious_hits:
        print(h.worker_id)

    suspicious_hits = verify_by_ip(hit_results, logs, ip_worker_matches)


def verify_by_logs():
    hit_scheme = get_ca4_scheme()
    common_dir = os.path.join(output_path, "ca_building", "run1", "mturk_output", "CA4")
    ca_shuffle_remain_A_path = os.path.join(common_dir, "shuffled_remain_A.csv")
    ca_shuffle_remain_B_path = os.path.join(common_dir, "shuffled_remain_B_master.csv")
    for path in [ca_shuffle_remain_A_path, ca_shuffle_remain_B_path]:
        hit_results: List[HitResult] = parse_file(path, hit_scheme)
        logs: List[ApacheLogParsed] = load_apache_log()
        suspicious_hits, ip_worker_matches = verify_by_time(hit_results, logs)
        print("{} of {} suspicious".format(len(suspicious_hits), len(logs)))



def show_agreement_after_drop():
    hit_scheme = get_ca4_scheme()
    hit_results: List[HitResult] = parse_file(sys.argv[1], hit_scheme)
    # foreach(apply_transitive, hit_results)
    show_agreement_inner_for_two(hit_results, cohens_kappa, scheme4_question_d, False)
    show_agreement_inner_for_two(hit_results, cohens_kappa, scheme4_question_d, True)


def show_agreement4():
    hit_scheme = get_ca4_scheme()
    hit_results: List[HitResult] = parse_file(sys.argv[1], hit_scheme)
    # hit_results = list([h for h in hit_results if h.worker_id != "A1PZ6FQ0WT3ROZ"])
    show_agreement_inner_for_two(hit_results, cohens_kappa, scheme4_question_d, False)


def get_ca3_ca_rate(input_path) -> SuccessCounter:
    hit_scheme = get_ca4_scheme()

    def get_binary_answer_from_hit_result(hit_result):
        answer_column = "Q8.on"
        worker_answer = hit_result.outputs[answer_column]
        return worker_answer == 1

    return get_ca_rate_common(input_path, hit_scheme, get_binary_answer_from_hit_result)


def check_if_same_guy_always_wrong():
    path_file_to_validate = sys.argv[1]
    hit_scheme = get_ca4_scheme()
    hit_results = parse_file(path_file_to_validate, hit_scheme)
    input_columns = list(hit_results[0].inputs.keys())
    answer_column = "Q8.on"

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


def answers():
    hit_scheme = get_ca4_scheme()
    hit_results: List[HitResult] = parse_file(sys.argv[1], hit_scheme)
    print(get_ca3_ca_rate(sys.argv[1]).get_suc_prob())
    worker_count = Counter([h.worker_id for h in hit_results])
    print(len(hit_results))

    for h in hit_results:
        if worker_count[h.worker_id] == 1:
            for key, value in h.outputs.items():
                h.outputs[key] = h.outputs[key] + 10

    answer_list_d = summarize_agreement(hit_results, min_entries=0)
    for answer_column, list_answers in answer_list_d.items():
        print(answer_column)
        print(list_answers)



def many_annot():
    hit_scheme = get_ca4_scheme()
    hit_results: List[HitResult] = parse_file(sys.argv[1], hit_scheme)
    worker_count = Counter([h.worker_id for h in hit_results])

    one_hit = list([h for h in hit_results if worker_count[h.worker_id] == 1])
    two_hit = list([h for h in hit_results if worker_count[h.worker_id] == 2])
    many_hit = list([h for h in hit_results if worker_count[h.worker_id] > 2])

    def get_counter_arg(h: HitResult):
        return h.outputs["Q8.on"]

    def print_ca_rate(h_list):
        answers = lmap(get_counter_arg, h_list)
        n_counter = sum(answers)
        rate = n_counter / len(answers)
        print("{} {} {}".format(n_counter, len(answers), rate))

    print_ca_rate(one_hit)
    print_ca_rate(two_hit)
    print_ca_rate(many_hit)


def group_by_claim():
    hit_scheme = get_ca4_scheme()
    hit_results: List[HitResult] = parse_file(sys.argv[1], hit_scheme)
    def get_claim(h: HitResult):
        return h.inputs['p_text']
    def get_doc_id(h: HitResult):
        return h.inputs['doc_id']

    groups = group_by(hit_results, get_claim)

    def get_counter_arg(h: HitResult):
        return h.outputs["Q8.on"]

    for key, items in groups.items():
        print(key)
        for doc_id, per_doc_items in group_by(items, get_doc_id).items():
            print(doc_id, lmap(get_counter_arg, per_doc_items))


def main():

    ##
    answers()


if __name__ == "__main__":
    main()
