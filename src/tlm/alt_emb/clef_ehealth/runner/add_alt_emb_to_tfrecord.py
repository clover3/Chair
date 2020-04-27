

import os

from cache import load_from_pickle
from cpath import output_path, pjoin
from misc_lib import exist_or_mkdir
from tlm.alt_emb.add_alt_emb import convert_alt_emb2


def convert(input_path, output_file_path):
    match_tree = load_from_pickle("match_tree_clef1_test")
    convert_alt_emb2(input_path, output_file_path, match_tree, True)


def main():
    in_dir = pjoin(output_path, "eHealth")
    exist_or_mkdir(in_dir)
    input_path_train = pjoin(in_dir, "tfrecord_train")
    input_path_test = pjoin(in_dir, "tfrecord_test")

    out_dir = os.path.join(output_path, "ehealth_alt")
    exist_or_mkdir(out_dir)
    output_file_path_train = os.path.join(out_dir, "train")
    output_file_path_test = os.path.join(out_dir, "test")

    convert(input_path_test, output_file_path_test)
    #convert(input_path_train, output_file_path_train)


if __name__ == "__main__":
    main()
