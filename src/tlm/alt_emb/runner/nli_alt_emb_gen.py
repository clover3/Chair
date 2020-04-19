import os

from cache import load_from_pickle
from cpath import output_path
from tlm.alt_emb.add_alt_emb import convert_alt_emb2


def convert(input_path, output_file_path):
    match_tree = load_from_pickle("match_tree_nli")
    convert_alt_emb2(input_path, output_file_path, match_tree, False)


def main():
    input_path = os.path.join(output_path, "nli_tfrecord_cls_300", "dev_mis")
    output_file_path = os.path.join(output_path, "nli_tfrecord_cls_300", "dev_mis_alt_small")
    convert(input_path, output_file_path)


if __name__ == "__main__":
    main()