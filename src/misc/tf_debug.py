from collections import OrderedDict
from typing import Iterable

import tensorflow as tf

from data_generator.tokenizer_wo_tf import get_tokenizer
from tlm.data_gen.classification_common import ClassificationInstance, encode_classification_instance

tf = tf.compat.v1
def main():
    tokens = ["hi", "hello"]
    seg_ids = [0,0,]
    inst = ClassificationInstance(tokens, seg_ids, 0)

    inst_list = [inst]

    out_path = "/tmp/temp.youngwoo"
    max_seq_length = 512
    tokenizer = get_tokenizer()
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    def encode_fn(inst: ClassificationInstance) -> OrderedDict:
        return encode_classification_instance(tokenizer, max_seq_length, inst)

    features_list: Iterable[OrderedDict] = map(encode_fn, inst_list)
    writer = tf.python_io.TFRecordWriter(out_path)
    for e in features_list:
        # features = OrderedDict()
        # features["input_ids"] = create_int_feature(input_ids)

        f = tf.train.Features(feature=e)
        tf_example = tf.train.Example(features=f)
        writer.write(tf_example.SerializeToString())



if __name__ == "__main__":
    main()
