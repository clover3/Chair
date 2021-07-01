import os
from collections import Counter
from typing import List

from arg.counter_arg_retrieval.build_dataset.run1.agreement_scheme2 import get_ca_run1_scheme2
from misc_lib import group_by
from mturk.parse_util import parse_file, HitResult


def count_counter_argument():
    common_dir = "C:\\work\\Code\\Chair\\output\\ca_building\\run1\\mturk_output"
    file_path = os.path.join(common_dir,  "Batch_4490355_batch_results.csv")
    hit_scheme = get_ca_run1_scheme2()
    hit_results: List[HitResult] = parse_file(file_path, hit_scheme)
    input_columns = list(hit_results[0].inputs.keys())

    def get_hit_key(hit_result: HitResult):
        return tuple([hit_result.inputs[key] for key in input_columns])
    answer_columns = list(hit_results[0].outputs.keys())

    counter = Counter()
    for key, entries in group_by(hit_results, get_hit_key).items():
        answer_column = "Q13"

        def get_answers_for_column(answer_column):
            return list([e.outputs[answer_column] for e in entries])

        for c in answer_columns:
            answers = get_answers_for_column(c)
            if sum(answers) >= 2:
                counter[c] += 1

        if len(entries) >= 2:
            counter['cnt'] += 1

    print(counter['cnt'])
    for key in answer_columns:
        print(key, counter[key])


def main():
    count_counter_argument()


if __name__ == "__main__":
    main()
