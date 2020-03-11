import os
import sys

import data_generator.argmining.ukp_header
from misc_lib import exist_or_mkdir
from tf_util.enum_features import load_record
from tf_util.record_writer_wrap import RecordWriterWrap
from tlm.data_gen.bert_data_gen import create_int_feature
from tlm.ukp.data_gen.add_topic_ids import feature_to_ordered_dict

token_ids_to_topic = {
    11324: "abortion",
    2082: "school_uniforms",
    4517: "nuclear_energy",
    16204: "marijuana_legalization",
    3282: "gun_control",
    6263: "minimum_wage",
    2331: "death_penalty",
    18856: "cloning",
}


def augment_topic_ids(records, save_path):
    writer = RecordWriterWrap(save_path)

    for feature in records:
        first_inst = feature_to_ordered_dict(feature)
        input_ids = first_inst["input_ids"].int64_list.value
        token_ids = input_ids[1]
        topic = token_ids_to_topic[token_ids]
        topic_id = data_generator.argmining.ukp_header.all_topics.index(topic)
        first_inst["topic_ids"] = create_int_feature([topic_id])
        writer.write_feature(first_inst)

    writer.close()



def run(dir_path, save_dir):
    exist_or_mkdir(save_dir)
    for split in ["train", "dev"]:
        for idx, topic in enumerate(data_generator.argmining.ukp_header.all_topics):
            file_name = "{}_{}".format(split, topic)
            file_path = os.path.join(dir_path, file_name)
            save_path = os.path.join(save_dir, file_name)
            augment_topic_ids(load_record(file_path), save_path)


if __name__ == '__main__':
    run(sys.argv[1], sys.argv[2])

