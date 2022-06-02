import sys
from typing import List

from arg.counter_arg_retrieval.build_dataset.run1.ca4_access_verify import get_ca4_scheme

from arg.counter_arg_retrieval.build_dataset.run1.annotation.scheme import get_ca_run1_scheme2
from arg.counter_arg_retrieval.build_dataset.verify_common import summarize_agreement, \
    get_agreement_rate_from_answer_list
from evals.agreement import cohens_kappa
from misc_lib import SuccessCounter
from mturk.parse_util import parse_file, HitResult


def true_rate(list_list):
    sc = SuccessCounter()
    for l in list_list:
        for e in l:
            if e:
                sc.suc()
            else:
                sc.fail()
    return sc


def combine_agreement_rate():
    hit_scheme = get_ca4_scheme()
    hit_results: List[HitResult] = parse_file(sys.argv[1], hit_scheme)
    answer_list_d = summarize_agreement(hit_results, min_entries=0)

    list_answers_4 = answer_list_d["Q8.on"]
    avg_k_4 = get_agreement_rate_from_answer_list(cohens_kappa, list_answers_4)

    hit_scheme = get_ca_run1_scheme2()
    hit_results: List[HitResult] = parse_file(sys.argv[2], hit_scheme)
    answer_list_d = summarize_agreement(hit_results, min_entries=0)
    list_answers_2 = answer_list_d["Q13.on"]
    avg_k_2 = get_agreement_rate_from_answer_list(cohens_kappa, list_answers_2)
    avg_k_combine = get_agreement_rate_from_answer_list(cohens_kappa, list_answers_4 + list_answers_2)

    for todo in [list_answers_2, list_answers_4, list_answers_2+list_answers_4]:
        avg_k = get_agreement_rate_from_answer_list(cohens_kappa, todo)
        sc = true_rate(todo)
        print(avg_k, sc.get_suc_prob(), sc.n_total)




def main():
    combine_agreement_rate()


if __name__ == "__main__":
    main()