from typing import NamedTuple

import tensorflow as tf

from tf_v2_support import placeholder
from tlm.model import base as bert


class InputStructureBase:
    def __init__(self, seq_length):
        self.input_ids = placeholder(tf.int64, [None, seq_length])
        self.input_mask = placeholder(tf.int64, [None, seq_length])
        self.segment_ids = placeholder(tf.int64, [None, seq_length])

    def get_input_ids(self):
        return self.input_ids

    def get_input_mask(self):
        return self.input_mask

    def get_segment_ids(self):
        return self.segment_ids

    def get_x_dict(self):
        d = dict({
            "input_ids": self.input_ids,
            "input_mask": self.input_mask,
            "segment_ids": self.segment_ids,
        })
        return d


class BertRunConfig(NamedTuple):
    use_one_hot_embeddings: bool
    is_training: bool
    bert_config: bert.BertConfig


class BertCommonHP:
    seq_max = 512  # Maximum number of words in a sentence. alias = T.
    hidden_units = 768  # alias = C
    num_blocks = 12  # number of encoder/decoder blocks
    num_heads = 12
    dropout_rate = 0.1
    sinusoid = False  # If True, use sinusoid. If false, positional embedding.
    intermediate_size = 3072
    type_vocab_size = 2
    voca_size = 30522


def get_bert_run_config(hp, is_training) -> BertRunConfig:
    config = bert.BertConfig(vocab_size=hp.voca_size,
                             hidden_size=hp.hidden_units,
                             num_hidden_layers=hp.num_blocks,
                             num_attention_heads=hp.num_heads,
                             intermediate_size=hp.intermediate_size,
                             type_vocab_size=hp.type_vocab_size,
                             )
    return BertRunConfig(False, is_training, config)