import sys
from collections import Counter, defaultdict
from typing import List

from arg.counter_arg_retrieval.build_dataset.run1.agreement_scheme2 import get_ca_run1_scheme2
from arg.counter_arg_retrieval.build_dataset.verify_common import get_list_of_known_labels
from misc_lib import get_dict_items
from mturk.parse_util import parse_file, HitResult


def check_wrong_annotation():
    hit_scheme = get_ca_run1_scheme2()
    hit_results: List[HitResult] = parse_file(sys.argv[1], hit_scheme)
    my_file_path = sys.argv[2]
    label_d = get_list_of_known_labels(my_file_path, hit_scheme.inputs, ['label'])
    hits_to_see = []
    count_match = Counter()
    count_notmatch = Counter()
    n_align = 0
    mturk_label_d = defaultdict(list)

    def convert_2_0(label_s):
        if label_s == "2":
            return 1
        if label_s == "1":
            return 0
        if label_s == "0":
            return 0

    for hit_result in hit_results:
        input_list = tuple(get_dict_items(hit_result.inputs, hit_scheme.inputs))
        if input_list in label_d:
            n_align += 1
            my_answer = convert_2_0(label_d[input_list][0])
            worker_answer = hit_result.outputs['Q13.on']
            mturk_label_d[input_list].append(worker_answer)

            if my_answer != worker_answer:
                print(input_list, my_answer, worker_answer)
                hits_to_see.append(input_list)
                count_notmatch[my_answer] += 1
            else:
                count_match[my_answer] += 1

    for key in mturk_label_d:
        print(label_d[key], mturk_label_d[key])
    print(count_match)
    print("{} of {} different".format(len(hits_to_see), n_align))


def main():
    return check_wrong_annotation()


if __name__ == "__main__":
    main()