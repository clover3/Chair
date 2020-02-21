import tensorflow as tf

from models.transformer.bert_common_v2 import get_shape_list
from tlm.model import base
from tlm.model.base import BertModel, create_initializer
from tlm.training.flags_wrapper import get_input_files_from_flags, show_input_files
from tlm.training.input_fn_common import format_dataset


class MultiContextEncoder(base.BertModelInterface):
    def __init__(self,
                 config, # This is different from BERT config,
                 is_training,
                 input_ids,
                 input_mask,
                 token_type_ids,
                 use_one_hot_embeddings,
                 features,
                 ):
        super(MultiContextEncoder, self).__init__()
        self.config = config
        if not is_training:
            config.set_attrib("hidden_dropout_prob", 0.0)
            config.set_attrib("attention_probs_dropout_prob", 0.0)

        def reform_context(context):
            return tf.reshape(context, [-1, config.max_context, config.max_context_length])

        batch_size, _ = get_shape_list(input_ids)
        def combine(input_ids, context_input_ids):
            a = tf.tile(tf.expand_dims(input_ids, 1), [1, config.max_context, 1])
            b = reform_context(context_input_ids)
            rep_3d = tf.concat([a, b], 2)
            return tf.reshape(rep_3d, [batch_size * config.max_context, -1])

        context_input_ids = features["context_input_ids"]
        context_input_mask = features["context_input_mask"]
        context_segment_ids = features["context_segment_ids"]
        context_segment_ids = tf.ones_like(context_segment_ids, tf.int32) * 2
        self.module = BertModel(config=config,
                                is_training=is_training,
                                input_ids=combine(input_ids, context_input_ids),
                                input_mask=combine(input_mask, context_input_mask),
                                token_type_ids=combine(token_type_ids, context_segment_ids),
                                use_one_hot_embeddings=use_one_hot_embeddings,
                                )
        dense_layer_setup = tf.keras.layers.Dense(config.hidden_size,
                                                  activation=tf.keras.activations.tanh,
                                                  kernel_initializer=create_initializer(config.initializer_range))
        h1 = self.module.get_pooled_output()
        h2 = dense_layer_setup(h1)
        h2 = tf.reshape(h2, [batch_size, config.max_context, -1])
        h2 = h2[:, :config.num_context]
        h3 = tf.reduce_mean(h2, axis=1)
        h4 = dense_layer_setup(h3)
        self.pooled_output = h4

    def get_pooled_output(self):
        return self.pooled_output


def input_fn_builder_multi_context_classification(max_seq_length, max_context, max_context_length, flags):
    input_files = get_input_files_from_flags(flags)
    show_input_files(input_files)
    is_training = flags.do_train
    num_cpu_threads = 4
    raw_context_len = max_context * max_context_length
    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]
        name_to_features = {
            "input_ids": tf.io.FixedLenFeature([max_seq_length], tf.int64),
            "input_mask": tf.io.FixedLenFeature([max_seq_length], tf.int64),
            "segment_ids": tf.io.FixedLenFeature([max_seq_length], tf.int64),
            "context_input_ids": tf.io.FixedLenFeature([raw_context_len], tf.int64),
            "context_input_mask": tf.io.FixedLenFeature([raw_context_len], tf.int64),
            "context_segment_ids": tf.io.FixedLenFeature([raw_context_len], tf.int64),
            "label_ids": tf.io.FixedLenFeature([1], tf.int64),
        }
        return format_dataset(name_to_features, batch_size, is_training, flags, input_files, num_cpu_threads)

    return input_fn
