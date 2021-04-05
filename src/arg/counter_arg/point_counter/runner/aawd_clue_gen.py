import json
import os
import sys
from typing import List, Dict, Tuple

from arg.counter_arg.point_counter.ada_gen import combine_source_and_target, get_encode_fn
from cpath import at_output_dir, output_path
from dataset_specific.aawd.load import load_aawd_splits_as_binary
from misc_lib import exist_or_mkdir
from tf_util.record_writer_wrap import write_records_w_encode_fn


def load_clue_unlabeled() -> List[Tuple[str, int]]:
    json_path = sys.argv[1]
    docs:  Dict[str, List[str]] = json.load(open(json_path, "r"))
    keys = list(docs.keys())
    print(keys[10])
    print(len(keys))
    doc_texts = docs.values()

    output = []
    char_per_window = 512 * 5
    for t in doc_texts:
        if len(t) > char_per_window:
            idx = 0
            while idx < len(t):
                e = t[idx:idx+char_per_window], 0
                output.append(e)
                idx += char_per_window
    return output


def main():
    aawd_train, _, _ = load_aawd_splits_as_binary()
    clue_unlabeld = load_clue_unlabeled()
    data_d = {
        'aawd': aawd_train,
        'clue': clue_unlabeld,
    }

    encode_fn = get_encode_fn(512)
    dir_name = "counter_argument_ada"
    exist_or_mkdir(os.path.join(output_path, dir_name))

    def make_tfrecord(source_name, target_name):
        source_data = data_d[source_name]
        target_data = data_d[target_name]
        combined_data = combine_source_and_target(source_data, target_data, 1)
        save_path = at_output_dir(dir_name, "{}_to_{}_train".format(source_name, target_name))
        write_records_w_encode_fn(save_path, encode_fn, combined_data)

    make_tfrecord("aawd", "clue")


if __name__ == "__main__":
    main()
