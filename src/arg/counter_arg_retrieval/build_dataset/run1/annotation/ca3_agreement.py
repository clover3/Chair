import sys
from typing import List, Any

from arg.counter_arg_retrieval.build_dataset.run1.annotation.scheme import get_ca3_scheme
from arg.counter_arg_retrieval.build_dataset.run1.enum_disagreeing_ones import load_ca3_master
from arg.counter_arg_retrieval.build_dataset.verify_common import summarize_agreement, annotator_eval, \
    show_agreement_inner_w_true_rate
from evals.agreement import cohens_kappa
from list_lib import lmap
from misc_lib import get_first, get_second, get_third
from mturk.parse_util import parse_file, HitResult

scheme3_question_d = {
    "Q1.": "claim supports topic",
    "Q2.": "claim opposes topic",
    "Q2": "why"
}


def avg_agreement():
    hit_results = load_ca3_master()
    # hit_scheme = get_ca3_scheme()
    # ca3_input = os.path.join(common_dir, "CA3", "Batch_4506547_batch_results.csv")
    # hit_results = parse_file(ca3_input, hit_scheme)
    # hit_results = list([h for h in hit_results if h.worker_id != "A1PZ6FQ0WT3ROZ"])
    show_agreement_inner_w_true_rate(hit_results, cohens_kappa, scheme3_question_d, False)



def show_agreement():
    hit_scheme = get_ca3_scheme()
    hit_results: List[HitResult] = parse_file(sys.argv[1], hit_scheme)
    answer_list_d = summarize_agreement(hit_results)
    for answer_column, list_answers in answer_list_d.items():
        if answer_column == 'key':
            continue
        annot1: List[Any] = lmap(get_first, list_answers)
        annot2: List[Any] = lmap(get_second, list_answers)
        annot3: List[Any] = lmap(get_third, list_answers)
        print("annot1", annot1)
        print("annot2", annot2)
        print("annot3", annot3)
        print(answer_column)
        print('1 vs 2', cohens_kappa(annot1, annot2))
        print('2 vs 3', cohens_kappa(annot2, annot3))
        print('3 vs 1', cohens_kappa(annot3, annot1))

    correct_counter, wrong_counter = annotator_eval(hit_results)

    for key in wrong_counter.keys():
        print(key, correct_counter[key], wrong_counter[key])


def main():
    avg_agreement()


if __name__ == "__main__":
    main()