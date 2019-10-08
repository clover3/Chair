import tensorflow as tf
import collections
from data_generator.tokenizer_b import FullTokenizer
import numpy as np
import path, os


def create_int_feature(values):
  feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
  return feature

def read_bert_data(fn):
    for record in tf.python_io.tf_record_iterator(fn):
        example = tf.train.Example()
        example.ParseFromString(record)
        feature = example.features.feature
        yield feature


class VocaMasker:
    def __init__(self, voca_path):
        self.tokenizer = FullTokenizer(voca_path)
        self.postfix_indice = set()
        self.max_seq = 512
        for sb in self.tokenizer.vocab:
            idx = self.tokenizer.vocab[sb]
            if sb.startswith("##"):
                self.postfix_indice.add(idx)

    def is_postfix(self, idx):
        return idx in self.postfix_indice

    def input_ids_to_attn_range(self, input_ids):
        last_begin = 0
        l = self.max_seq
        intervals = []
        for i, token_id in enumerate(input_ids):
            if not self.is_postfix(token_id):
                intervals.append((last_begin, i))
                last_begin = i

        intervals.append((last_begin, self.max_seq))

        attention_range = []
        for st, ed in intervals:
            for i in range(st, ed):
                attention_range.append(ed)
        assert len(attention_range) == self.max_seq
        return attention_range


def convert_write(output_file, examples):
    vocab_file = os.path.join(path.data_path, "bert_voca.txt")
    vm = VocaMasker(vocab_file)
    writers = tf.python_io.TFRecordWriter(output_file)
    cnt =0
    for feature in examples:
        new_feature = collections.OrderedDict()
        for key in feature.keys():
            new_feature[key] = feature[key]

        attn_range = vm.input_ids_to_attn_range(feature['input_ids'].int64_list.value)
        new_feature["voca_mask"] = create_int_feature(attn_range)
        tf_example = tf.train.Example(features=tf.train.Features(feature=new_feature))
        writers.write(tf_example.SerializeToString())

def dev():
    path = "/mnt/nfs/work3/youngwookim/data/bert_tf/tf/done/0"
    out_path = "/mnt/nfs/work3/youngwookim/data/ibert_tf/0"
    data = read_bert_data(path)
    convert_write(out_path, data)

if __name__ == "__main__":
    dev()

