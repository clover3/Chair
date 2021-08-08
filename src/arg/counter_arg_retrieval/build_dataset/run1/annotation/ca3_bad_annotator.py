
import sys
from collections import Counter, defaultdict

from arg.counter_arg_retrieval.build_dataset.run1.annotation.scheme import get_ca3_scheme
from arg.counter_arg_retrieval.build_dataset.verify_common import get_list_of_known_labels
from misc_lib import get_dict_items
from mturk.parse_util import parse_file, remove_rejected


def main():
    my_file_path = sys.argv[1]
    path_file_to_validate = sys.argv[2]
    hit_scheme = get_ca3_scheme()
    label_d = get_list_of_known_labels(my_file_path, hit_scheme.inputs, ['label'])
    print("{} known labels".format(len(label_d)))
    hit_results = parse_file(path_file_to_validate, hit_scheme)

    hit_results = remove_rejected(hit_results)

    hits_to_see = []
    count_match = Counter()
    count_notmatch = Counter()
    mturk_label_d = defaultdict(list)
    n_align = 0
    for hit_result in hit_results:
        input_list = tuple(get_dict_items(hit_result.inputs, hit_scheme.inputs))
        if input_list in label_d:
            n_align += 1
            my_answer = int(label_d[input_list][0])
            worker_answer = hit_result.outputs['Q2.']
            mturk_label_d[input_list].append(worker_answer)

            if my_answer != worker_answer:
                print(my_answer, worker_answer)
                count_notmatch[my_answer] += 1
                hits_to_see.append(input_list)
            else:
                count_match[my_answer] += 1

    for key in mturk_label_d:
        print(key, label_d[key], mturk_label_d[key])

    print("{} of {} different".format(len(hits_to_see), n_align))
    print("my label when matched", count_match)
    print("my label when not matched", count_notmatch)


if __name__ == "__main__":
    main()