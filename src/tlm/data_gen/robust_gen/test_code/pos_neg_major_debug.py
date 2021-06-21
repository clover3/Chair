import sys
from collections import Counter

from misc_lib import get_dir_files
from tf_util.enum_features import load_record
from tlm.data_gen.feature_to_text import take


def main():
    dir_path = sys.argv[1]
    file_itr = get_dir_files(dir_path)
    for file_path in file_itr:
        pos_counter = Counter()
        neg_counter = Counter()
        for feature in load_record(file_path):
            def get_hash(feature_name):
                input_ids = take(feature[feature_name])
                input_hash = hash(str(list(input_ids)))
                return input_hash

            pos_counter[get_hash("input_ids1")] += 1
            neg_counter[get_hash("input_ids2")] += 1

        def count_frequency(counter):
            cnt_counter = Counter()
            for key in counter:
                cnt = counter[key]
                cnt_counter[cnt] += 1
            return cnt_counter

        def print_counter_count(name, counter):
            print("{} {}".format(name, count_frequency(counter)))

        print_counter_count("pos", pos_counter)
        print_counter_count("neg", neg_counter)


if __name__ == "__main__":
    main()