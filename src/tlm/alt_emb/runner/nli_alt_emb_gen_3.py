

import os

from cache import load_from_pickle
from cpath import output_path
from misc_lib import exist_or_mkdir
from tlm.alt_emb.add_alt_emb import convert_alt_emb2


def convert(input_path, output_file_path):
    match_tree = load_from_pickle("match_tree_nli_dev")
    convert_alt_emb2(input_path, output_file_path, match_tree, True)


def main():
    input_path_train = os.path.join(output_path, "nli_tfrecord_cls_300", "train")
    input_path_dev = os.path.join(output_path, "nli_tfrecord_cls_300", "dev")
    input_path_dev_mis = os.path.join(output_path, "nli_tfrecord_cls_300", "dev_mis")
    out_dir = os.path.join(output_path, "nli_tfrecord_cls_300_alt_3")
    exist_or_mkdir(out_dir)
    output_file_path_train = os.path.join(out_dir, "train")
    output_file_path_dev = os.path.join(out_dir, "dev")
    output_file_path_dev_mis = os.path.join(out_dir, "dev_mis")

    convert(input_path_dev, output_file_path_dev)
    convert(input_path_dev_mis, output_file_path_dev_mis)
    convert(input_path_train, output_file_path_train)


if __name__ == "__main__":
    main()