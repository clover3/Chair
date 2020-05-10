


import os

from cache import load_from_pickle
from cpath import output_path, pjoin, data_path
from data_generator.argmining.ukp_header import all_topics
from misc_lib import exist_or_mkdir
from tlm.alt_emb.add_alt_emb import convert_alt_emb2


def convert(input_path, output_file_path):
    match_tree = load_from_pickle("match_tree_ukp")
    convert_alt_emb2(input_path, output_file_path, match_tree, True)


def main():
    dataset_dir = pjoin(data_path, "ukp_300")

    for topic in all_topics:
        train_data_path = pjoin(dataset_dir, "train_{}".format(topic))
        test_data_path = pjoin(dataset_dir, "dev_{}".format(topic))

        out_dir = os.path.join(output_path, "ukp_alt")
        exist_or_mkdir(out_dir)
        output_file_path_train = os.path.join(out_dir, "train_{}".format(topic))
        output_file_path_test = os.path.join(out_dir, "dev_{}".format(topic))

        convert(test_data_path, output_file_path_test)
        convert(train_data_path, output_file_path_train)


if __name__ == "__main__":
    main()
