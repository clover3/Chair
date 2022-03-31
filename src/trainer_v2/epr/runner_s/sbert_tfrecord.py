import collections
import os
from collections import OrderedDict

import tensorflow as tf
from transformers import MPNetTokenizer

from cpath import output_path
from data_generator.create_feature import create_int_feature
from misc_lib import exist_or_mkdir
from tf_util.record_writer_wrap import RecordWriterWrap
from trainer_v2.epr.s_bert_enc import load_segmented_data


def create_string_feature(str_list):
    bytes_list = [s.encode("utf-8") for s in str_list]
    feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=bytes_list))
    return feature


def encode(tokenizer, d) -> OrderedDict:
    features = collections.OrderedDict()
    for seg_name in ["premise", "hypothesis"]:
        f_d = tokenizer(d[seg_name])
        # print(f_d.keys())
        for sub_name in ["input_ids", "attention_mask"]:
            ids_list_list = f_d[sub_name]
            tensor = tf.ragged.constant(ids_list_list).to_tensor()
            serialized_nonscalar = tf.io.serialize_tensor(tensor)
            feature = tf.train.Feature(bytes_list=tf.train.BytesList(
                value=[serialized_nonscalar.numpy()]))
            features[f"{seg_name}_{sub_name}"] = feature
    features['label'] = create_int_feature([d['label']])
    return features


def convert():
    tokenizer = MPNetTokenizer.from_pretrained("microsoft/mpnet-base")
    dataset = "snli"
    split = "validation"
    job_no = 0
    json_iter = load_segmented_data(dataset, split, job_no)
    dir_name = f"{dataset}_{split}"
    out_dir = os.path.join(output_path, "epr", dir_name)
    exist_or_mkdir(out_dir)
    out_file = os.path.join(out_dir, str(job_no))
    writer = RecordWriterWrap(out_file)
    for item in json_iter:
        writer.write_feature(encode(tokenizer, item))


if __name__ == "__main__":
    convert()