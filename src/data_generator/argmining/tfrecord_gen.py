import os
import random

from cpath import data_path
from data_generator.argmining.ukp import BertDataLoader
from data_generator.argmining.ukp_header import all_topics
from data_generator.tfrecord_gen import modify_data_loader, entry_to_feature_dict, write_features_to_file, \
    pairwise_entry_to_feature_dict
from list_lib import lmap
from misc_lib import exist_or_mkdir
from tlm.data_gen.pairwise_common import generate_pairwise_combinations


def gen_tfrecord():
    max_sequence = 512
    dir_path = os.path.join(data_path, "ukp_{}".format(max_sequence))
    exist_or_mkdir(dir_path)
    for topic in all_topics:
        data_loader = modify_data_loader(BertDataLoader(topic, True, max_sequence, "bert_voca.txt", "only_topic_word"))
        todo = [("train", data_loader.get_train_data()), ("dev", data_loader.get_dev_data())]

        for name, data in todo[::-1]:
            features = lmap(entry_to_feature_dict, data)
            out_name = "{}_{}".format(name, topic)
            out_path = os.path.join(dir_path, out_name)
            write_features_to_file(features, out_path)


def gen_pairwise():
    max_sequence = 300
    dir_path = os.path.join(data_path, "ukp_pairwise_{}".format(max_sequence))
    exist_or_mkdir(dir_path)

    for topic in all_topics:
        data_loader = modify_data_loader(BertDataLoader(topic, True, max_sequence, "bert_voca.txt", "only_topic_word"))
        todo = [("train", data_loader.get_train_data()), ("dev", data_loader.get_dev_data())]
        for name, data in todo[::-1]:
            out_name = "{}_{}".format(name, topic)
            out_path = os.path.join(dir_path, out_name)

            grouped = [[], [], []]
            for e in data:
                input_ids, input_mask, segment_ids, label = e
                grouped[label].append(e)

            combs = []
            combs.extend(generate_pairwise_combinations(grouped[0], grouped[1]))
            combs.extend(generate_pairwise_combinations(grouped[1], grouped[2]))
            combs.extend(generate_pairwise_combinations(grouped[2], grouped[0]))
            features = lmap(pairwise_entry_to_feature_dict, combs)
            write_features_to_file(features, out_path)


def gen_tfrecord_w_tdev():
    max_sequence = 300
    dir_path = os.path.join(data_path, "ukp_tdev_{}".format(max_sequence))
    exist_or_mkdir(dir_path)
    for topic in all_topics:
        data_loader = modify_data_loader(BertDataLoader(topic, True, max_sequence, "bert_voca.txt", "only_topic_word"))
        todo = [("dev", data_loader.get_dev_data())]

        train_data = list(data_loader.get_train_data())

        random.shuffle(train_data)
        validation_size = int(len(train_data) * 0.1)

        train_train_data = train_data[:-validation_size]
        train_dev_data = train_data[validation_size:]
        todo.append(("ttrain", train_train_data))
        todo.append(("tdev", train_dev_data))

        for name, data in todo[::-1]:
            features = lmap(entry_to_feature_dict, data)
            out_name = "{}_{}".format(name, topic)
            out_path = os.path.join(dir_path, out_name)
            write_features_to_file(features, out_path)


if __name__ == "__main__":
    gen_pairwise()
