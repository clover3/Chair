from typing import Callable, List, Tuple

import tensorflow as tf
from transformers import AutoTokenizer


def get_qd_encoder(max_seq_length) -> Callable[[List[Tuple[str, str]]], tf.data.Dataset]:
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    SpecI = tf.TensorSpec([max_seq_length], dtype=tf.int32)
    sig = (SpecI, SpecI, ),

    def encode_qd(qd_list) -> tf.data.Dataset:
        def generator():
            for query, document in qd_list:
                encoded_input = tokenizer.encode_plus(
                    query,
                    document,
                    padding="max_length",
                    max_length=max_seq_length,
                    truncation=True,
                    return_tensors="tf"
                )
                yield (encoded_input['input_ids'][0], encoded_input['token_type_ids'][0]),

        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=sig)
        return dataset
    return encode_qd


def get_dummy_input_for_bert_layer():
    DUMMY_INPUTS = [[7, 6, 0, 0, 1], [1, 2, 3, 0, 0], [0, 0, 0, 4, 5]]
    dummy = {"input_ids": tf.constant(DUMMY_INPUTS, dtype=tf.int32)}
    return dummy