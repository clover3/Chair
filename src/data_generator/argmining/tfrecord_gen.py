import os

from cpath import data_path
from data_generator.argmining.ukp import BertDataLoader, all_topics, lmap
from data_generator.tfrecord_gen import modify_data_loader, entry_to_feature_dict, write_features_to_file
from misc_lib import exist_or_mkdir


def gen_tfrecord():
    max_sequence = 300
    dir_path = os.path.join(data_path, "ukp_{}".format(max_sequence))
    exist_or_mkdir(dir_path)
    for topic in all_topics:
        data_loader = modify_data_loader(BertDataLoader(topic, True, max_sequence, "bert_voca.txt", "only_topic_word"))
        todo = [("train", data_loader.get_train_data()), ("dev", data_loader.get_dev_data())]

        for name, data in todo:
            features = lmap(entry_to_feature_dict, data)
            out_name = "{}_{}".format(name, topic)
            out_path = os.path.join(dir_path, out_name)
            write_features_to_file(features, out_path)


if __name__ == "__main__":
    gen_tfrecord()
