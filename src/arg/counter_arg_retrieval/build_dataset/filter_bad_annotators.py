import csv
import sys
from collections import Counter, defaultdict
from typing import List, Iterable, Dict

from arg.counter_arg_retrieval.build_dataset.summarize_mturk import get_ca_run1_scheme
from misc_lib import get_dict_items
from mturk.parse_util import parse_file, HitResult


def get_column_values(columns, row, row_name_to_loc):
    input_keys = []
    for key in columns:
        input_keys.append(row[row_name_to_loc[key]])
    return input_keys


def get_list_of_known_labels(path, inputs: Iterable[str], answers: Iterable[str]) -> Dict:
    f = open(path, "r", encoding="utf-8")
    out_d = {}
    row_name_to_loc = {}
    for idx, row in enumerate(csv.reader(f)):
        if idx == 0:
            for row_idx, item in enumerate(row):
                row_name_to_loc[item] = row_idx
        else:
            input_values = get_column_values(inputs, row, row_name_to_loc)
            output_values = get_column_values(answers, row, row_name_to_loc)
            out_d[tuple(input_values)] = output_values
    return out_d

def remove_rejected(hit_results: List[HitResult]):
    output = []
    for h in hit_results:
        if h.status != "Rejected":
            output.append(h)
    return output


def main():
    my_file_path = sys.argv[1]
    path_file_to_validate = sys.argv[2]
    hit_scheme = get_ca_run1_scheme()
    label_d = get_list_of_known_labels(my_file_path, hit_scheme.inputs, ['label'])
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
            worker_answer = hit_result.outputs['relevant.label']
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
