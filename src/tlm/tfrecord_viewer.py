import tensorflow as tf
import os
from path import data_path
from data_generator import tokenizer_wo_tf
import sys

def read_bert_data(fn):
    for record in tf.python_io.tf_record_iterator(fn):
        example = tf.train.Example()
        example.ParseFromString(record)
        feature = example.features.feature
        yield feature


def inst2str(feature, tokenizer):
    def pretty(t):
        if t[:2] == "##":
            return t[2:]
        else:
            return t

    a = feature["input_ids"].int64_list.value

    mask_idx = 0
    mask_tokens = tokenizer.convert_ids_to_tokens(feature["masked_lm_ids"].int64_list.value)
    masked_positions = feature["masked_lm_positions"].int64_list.value
    out_str = ""
    for i, t in enumerate(tokenizer.convert_ids_to_tokens(a)):
        if t == "[PAD]":
            break

        skip_space = t[:2] == "##"
        if not skip_space:
            out_str += " "

        t = pretty(t)
        if i in masked_positions:
            out_str += "({}={})".format(t, mask_tokens[mask_idx])
            mask_idx += 1
        else:
            out_str += t
    return out_str

def read(fn):
    examples = read_bert_data(fn)
    tokenizer = tokenizer_wo_tf.FullTokenizer(os.path.join(data_path, "bert_voca.txt"))

    for feature in examples:
        print(inst2str(feature, tokenizer))
        print()
        print()


if __name__ == "__main__":
    fn = sys.argv[1]
    read(fn)