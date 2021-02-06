import tensorflow as tf

# MES: Multi-Evidence Single input
from models.transformer.bert_common_v2 import get_shape_list2, create_initializer
from tlm.model.base import BertModel, BertModelInterface
from tlm.model.multiple_evidence import split_input, r3to2


def get_combiner(is_training, config):
    option = config.modeling_option
    num_window = config.num_window

    def window_wise_dropout(first_tokens):
        if is_training:
            first_tokens = tf.nn.dropout(first_tokens, rate=1 / num_window, noise_shape=[1, num_window, 1], seed=1)
        return first_tokens

    # seq_output : [batch_size, num_window, seq_length, hidden_size]
    # output : [batch_size, hidden_size]
    def mlp_max(seq_output):
        # first_tokens : [batch_size, num_window, hidden_size]
        first_tokens = seq_output[:, :, 0, :]
        dense_layer1 = tf.keras.layers.Dense(config.hidden_size * 4,
                                            activation=tf.keras.activations.tanh,
                                            kernel_initializer=create_initializer(config.initializer_range))
        dense_layer2 = tf.keras.layers.Dense(config.hidden_size,
                                             activation=tf.keras.activations.tanh,
                                             kernel_initializer=create_initializer(config.initializer_range))

        first_tokens = window_wise_dropout(first_tokens)

        # hidden1 : [batch_size, num_window, hidden_size]
        hidden1 = dense_layer1(first_tokens)
        # hidden1 : [batch_size, num_window, hidden_size
        hidden2 = dense_layer2(hidden1)
        return tf.reduce_max(hidden2, axis=1)

    def direct_max(seq_output):
        # first_tokens : [batch_size, num_window, hidden_size]
        first_tokens = seq_output[:, :, 0, :]
        first_tokens = window_wise_dropout(first_tokens)

        return tf.reduce_max(first_tokens, axis=1)

    def use_first(seq_output):
        first_tokens = seq_output[:, :, 0, :]
        return first_tokens[:, 0, :]

    name_to_func = {
        'mlp_max': mlp_max,
        'direct_max': direct_max,
        'use_first': use_first,
    }
    return name_to_func[option]


# Final probability is built from the combination of vectors
class MES(BertModelInterface):
    def __init__(self,
                 config,
                 is_training,
                 input_ids,
                 input_mask=None,
                 token_type_ids=None,
                 use_one_hot_embeddings=True,
                 features=None,
                 scope=None):
        super(MES, self).__init__()
        combiner = get_combiner(is_training, config)

        unit_length = config.max_seq_length
        d_seq_length = config.max_d_seq_length
        num_window = int(d_seq_length / unit_length)
        batch_size, _ = get_shape_list2(input_ids)

        def dense(hidden_size, name):
            return tf.keras.layers.Dense(hidden_size,
                                         activation=tf.keras.activations.tanh,
                                         name=name,
                                         kernel_initializer=create_initializer(config.initializer_range))

        def r2to3(arr):
            return tf.reshape(arr, [batch_size, num_window, -1])

        def r3to4(arr):
            return tf.reshape(arr, [batch_size, num_window, unit_length, -1])

        def get_seq_output_3d(model_class, input_ids, input_masks, segment_ids):
            # [Batch, num_window, unit_seq_length]
            stacked_input_ids, stacked_input_mask, stacked_segment_ids = split_input(input_ids,
                                                                                     input_masks,
                                                                                     segment_ids,
                                                                                     d_seq_length,
                                                                                     unit_length)
            model = model_class(
                config=config,
                is_training=is_training,
                input_ids=r3to2(stacked_input_ids),
                input_mask=r3to2(stacked_input_mask),
                token_type_ids=r3to2(stacked_segment_ids),
                use_one_hot_embeddings=use_one_hot_embeddings,
            )

            # [Batch * num_window, seq_length, hidden_size]
            sequence = model.get_sequence_output()
            # [Batch, num_window, window_length, hidden_size]
            return r3to4(sequence)

        segment_ids = token_type_ids

        # [Batch, num_window, window_length, hidden_size]
        seq_output = get_seq_output_3d(BertModel, input_ids, input_mask, segment_ids)
        print(seq_output)

        self.pooled_output = combiner(seq_output)


# Final probability is mean of each segments
class MES_prob(BertModelInterface):
    def __init__(self,
                 config,
                 is_training,
                 use_one_hot_embeddings=True,
                 features=None,
                 scope=None):

        super(MES_prob, self).__init__()
        if config.modeling_option == "reduce_mean":
            def combine_probs(probs):
                return tf.reduce_mean(probs, axis=1)
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]

        unit_length = config.max_seq_length
        d_seq_length = config.max_d_seq_length
        num_window = int(d_seq_length / unit_length)
        batch_size, _ = get_shape_list2(input_ids)

        def dense(hidden_size, name):
            return tf.keras.layers.Dense(hidden_size,
                                         activation=tf.keras.activations.tanh,
                                         name=name,
                                         kernel_initializer=create_initializer(config.initializer_range))

        def r2to3(arr):
            return tf.reshape(arr, [batch_size, num_window, -1])

        def r3to4(arr):
            return tf.reshape(arr, [batch_size, num_window, unit_length, -1])

        def get_seq_output_3d(model_class, input_ids, input_masks, segment_ids):
            # [Batch, num_window, unit_seq_length]
            stacked_input_ids, stacked_input_mask, stacked_segment_ids = split_input(input_ids,
                                                                                     input_masks,
                                                                                     segment_ids,
                                                                                     d_seq_length,
                                                                                     unit_length)
            model = model_class(
                config=config,
                is_training=is_training,
                input_ids=r3to2(stacked_input_ids),
                input_mask=r3to2(stacked_input_mask),
                token_type_ids=r3to2(stacked_segment_ids),
                use_one_hot_embeddings=use_one_hot_embeddings,
            )

            # [Batch * num_window, seq_length, hidden_size]
            sequence = model.get_sequence_output()
            # [Batch, num_window, window_length, hidden_size]
            return r3to4(sequence)

        # [Batch, num_window, window_length, hidden_size]
        seq_output = get_seq_output_3d(BertModel, input_ids, input_mask, segment_ids)
        print(seq_output)
        first_token = seq_output[:, :, 0, :]
        hidden1 = dense(config.hidden_size, "hidden1")(first_token)
        pooled = dense(config.hidden_size, "hidden2")(hidden1)
        logits = tf.keras.layers.Dense(2, name="cls_dense")(pooled)
        probs = tf.nn.softmax(logits)[:, :, 1]  # [batch_size, num_window]

        self.prob = combine_probs(probs)

    def get_prob(self):
        return self.prob


class MES_loss(BertModelInterface):
    def __init__(self,
                 config,
                 is_training,
                 use_one_hot_embeddings=True,
                 features=None,
                 scope=None):
        super(MES_loss, self).__init__()
        if config.modeling_option == "reduce_mean":
            def combine_probs(probs):
                return tf.reduce_mean(probs, axis=1)
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]

        unit_length = config.max_seq_length
        d_seq_length = config.max_d_seq_length
        num_window = int(d_seq_length / unit_length)
        batch_size, _ = get_shape_list2(input_ids)

        def dense(hidden_size, name):
            return tf.keras.layers.Dense(hidden_size,
                                         activation=tf.keras.activations.tanh,
                                         name=name,
                                         kernel_initializer=create_initializer(config.initializer_range))

        def r2to3(arr):
            return tf.reshape(arr, [batch_size, num_window, -1])

        def r3to4(arr):
            return tf.reshape(arr, [batch_size, num_window, unit_length, -1])

        def get_seq_output_3d(model_class, input_ids, input_masks, segment_ids):
            # [Batch, num_window, unit_seq_length]
            stacked_input_ids, stacked_input_mask, stacked_segment_ids = split_input(input_ids,
                                                                                     input_masks,
                                                                                     segment_ids,
                                                                                     d_seq_length,
                                                                                     unit_length)
            model = model_class(
                config=config,
                is_training=is_training,
                input_ids=r3to2(stacked_input_ids),
                input_mask=r3to2(stacked_input_mask),
                token_type_ids=r3to2(stacked_segment_ids),
                use_one_hot_embeddings=use_one_hot_embeddings,
            )

            # [Batch * num_window, seq_length, hidden_size]
            sequence = model.get_sequence_output()
            # [Batch, num_window, window_length, hidden_size]
            return r3to4(sequence)
        seq_output = get_seq_output_3d(BertModel, input_ids, input_mask, segment_ids)
        print(seq_output)
        first_token = seq_output[:, :, 0, :]
        hidden1 = dense(config.hidden_size, "hidden1")(first_token)
        pooled = dense(config.hidden_size, "hidden2")(hidden1)
        logits = tf.keras.layers.Dense(2, name="cls_dense")(pooled)
        probs = tf.nn.softmax(logits)[:, :, 1]  # [batch_size, num_window]

        self.prob = combine_probs(probs)


    def get_loss(self):
        y_true = tf.cast(label_ids, tf.float32)
        # y_true = tf.cast(tf.one_hot(label_ids, 2), tf.float32)
        loss_arr = tf.keras.losses.BinaryCrossentropy()(y_true, probs)
        # loss_arr = tf.keras.losses.categorical_crossentropy(y_true, prob2d, from_logits=False,)

        loss = tf.reduce_mean(input_tensor=loss_arr)


