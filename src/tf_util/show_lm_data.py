import sys

import tensorflow as tf

from data_generator.common import get_tokenizer
from data_generator.tokenizer_wo_tf import pretty_tokens


def file_show(fn):
    tokenizer = get_tokenizer()
    for record in tf.compat.v1.python_io.tf_record_iterator(fn):
        example = tf.train.Example()
        example.ParseFromString(record)
        feature = example.features.feature
        keys = feature.keys()

        for key in keys:
            if key == "input_ids":
                v = feature[key].int64_list.value
                print(pretty_tokens(tokenizer.convert_ids_to_tokens(v), True))
        break

if __name__ == "__main__":
    file_show(sys.argv[1])
