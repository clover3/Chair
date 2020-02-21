import collections
import os
import sys

from data_generator.argmining import ukp
from tf_util.enum_features import load_record
from tf_util.record_writer_wrap import RecordWriterWrap
from tlm.data_gen.bert_data_gen import create_int_feature
from tlm.dictionary.feature_to_text import take


def feature_to_ordered_dict(feature):
    new_features = collections.OrderedDict()
    for key in feature:
        new_features[key] = create_int_feature(take(feature[key]))
    return new_features


def augment_topic_ids(records, topic_id, save_path):
    writer = RecordWriterWrap(save_path)
    for feature in records:
        first_inst = feature_to_ordered_dict(feature)
        first_inst["topic_ids"] = create_int_feature([topic_id])
        writer.write_feature(first_inst)

    writer.close()


def run(dir_path, save_dir):
    for idx, topic in enumerate(ukp.all_topics):
        file_path = os.path.join(dir_path, topic)
        save_path = os.path.join(save_dir, topic)
        augment_topic_ids(load_record(file_path), idx, save_path)


if __name__ == '__main__':
    run(sys.argv[1], sys.argv[2])

