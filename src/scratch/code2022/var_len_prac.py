import time

import tensorflow as tf

from data_generator.tokenizer_wo_tf import get_tokenizer
from dataset_specific.mnli.mnli_reader import MNLIReader
from trainer_v2.custom_loop.modeling_common.var_len_helper import concat, get_pad_fn


def enum_var_items(split):
    reader = MNLIReader()
    tokenizer = get_tokenizer()
    max_seq_length = 300
    tokens1 = None
    for e in reader.load_split(split):
        if tokens1 is None:
            tokens1 = tokenizer.tokenize(e.premise)
        # tokens2 = tokenizer.tokenize(e.hypothesis)
        tokens2 = tokens1
        ids1 = tokenizer.convert_tokens_to_ids(tokens1)
        ids2 = tokenizer.convert_tokens_to_ids(tokens2)
        yield ids1, ids2


def main():
    int_list = tf.TensorSpec(shape=(None,), dtype=tf.int32)
    output_signature = (int_list, int_list)
    dataset = tf.data.Dataset.from_generator(
        lambda: enum_var_items("dev"), output_signature=output_signature)

    max_seq_length = 300
    do_pad = get_pad_fn(max_seq_length)

    def pad_pair(a, b):
        return do_pad(a), do_pad(b)

    dataset = dataset.map(concat)
    dataset = dataset.map(pad_pair)
    dataset = dataset.batch(16)
    print("Starting")
    st = time.time()
    output = []
    for item in iter(dataset):
        a, b = item
        output.append(tf.reduce_sum(a))

    ed = time.time()
    print(len(output))
    print(ed - st)


if __name__ == "__main__":
    main()
