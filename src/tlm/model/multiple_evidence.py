import tensorflow as tf

from tlm.model.base import BertModelInterface, BertModel, create_initializer, get_shape_list2
from tlm.model.dual_model_common import *


def split_input(input_ids,
                input_mask,
                segment_ids,
                total_seq_length: int,
                window_length: int,
                ):
    num_window = int(total_seq_length / window_length)
    batch_size, _ = get_shape_list2(input_ids)

    def r2to3(arr):
        return tf.reshape(arr, [batch_size, num_window, -1])

    stacked_input_ids = r2to3(input_ids)  # [batch_size, num_window, src_window_length]
    stacked_input_mask = r2to3(input_mask)  # [batch_size, num_window, src_window_length]
    stacked_segment_ids = r2to3(segment_ids)  # [batch_size, num_window, src_window_length]
    return stacked_input_ids, stacked_input_mask, stacked_segment_ids


def r3to2(t):
    a, b, c = get_shape_list2(t)
    return tf.reshape(t, [-1, c])


class MultiEvidence(BertModelInterface):
    def __init__(self,
                 config,
                 is_training,
                 input_ids,
                 input_mask=None,
                 token_type_ids=None,
                 use_one_hot_embeddings=True,
                 features=None,
                 scope=None):
        super(MultiEvidence, self).__init__()

        input_ids2 = features["input_ids2"]
        input_mask2 = features["input_mask2"]
        segment_ids2 = features["segment_ids2"]

        input_ids3 = features["input_ids3"]
        input_mask3 = features["input_mask3"]
        segment_ids3 = features["segment_ids3"]

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

        def get_pooled_output_3d(model_class, input_ids, input_masks, segment_ids):
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

            # [Batch * num_window, hidden_size]
            raw_pooled = model.get_pooled_output()
            # [Batch, num_window, hidden_size]
            pooled_3d = r2to3(raw_pooled)
            return pooled_3d

        with tf.compat.v1.variable_scope(triple_model_prefix1):
            model_1 = BertModel(
                config=config,
                is_training=is_training,
                input_ids=input_ids,
                input_mask=input_mask,
                token_type_ids=token_type_ids,
                use_one_hot_embeddings=use_one_hot_embeddings,
            )

        with tf.compat.v1.variable_scope(triple_model_prefix2):
            pooled2 = get_pooled_output_3d(BertModel, input_ids2, input_mask2, segment_ids2)

        with tf.compat.v1.variable_scope(triple_model_prefix3):
            pooled3 = get_pooled_output_3d(BertModel, input_ids3, input_mask3, segment_ids3)

        per_doc_vector = tf.concat([pooled2, pooled3], axis=2)

        hidden1 = dense(config.hidden_size, "hidden1")(per_doc_vector)
        hidden2 = dense(config.hidden_size, "hidden2")(hidden1)

        combined = tf.reduce_max(hidden2, axis=1)

        model_1_first_token = model_1.get_sequence_output()[:, 0, :]

        rep = tf.concat([model_1_first_token, combined], axis=1)

        self.sequence_output = model_1.get_sequence_output()  # dummy
        dense_layer = tf.keras.layers.Dense(config.hidden_size,
                                            name="hidden3",
                                            activation=tf.keras.activations.tanh,
                                            kernel_initializer=create_initializer(config.initializer_range))
        pooled_output = dense_layer(rep)
        self.pooled_output = pooled_output


class MultiEvidenceUseSecond(BertModelInterface):
    def __init__(self,
                 config,
                 is_training,
                 input_ids,
                 input_mask=None,
                 token_type_ids=None,
                 use_one_hot_embeddings=True,
                 features=None,
                 scope=None):
        super(MultiEvidenceUseSecond, self).__init__()

        input_ids2 = features["input_ids2"]
        input_mask2 = features["input_mask2"]
        segment_ids2 = features["segment_ids2"]

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
            # [Batch, num_window, hidden_size]
            return r3to4(sequence)

        stacked_input_ids, stacked_input_mask, stacked_segment_ids = split_input(input_ids2,
                                                                                 input_mask2,
                                                                                 segment_ids2,
                                                                                 d_seq_length,
                                                                                 unit_length)

        seq_output = get_seq_output_3d(BertModel, input_ids2, input_mask2, segment_ids2)
        is_first_seg = tf.logical_and(tf.equal(stacked_segment_ids, 0), tf.equal(stacked_input_mask, 1))
        is_first_seg_f = tf.expand_dims(tf.cast(is_first_seg, tf.float32), -1)
        masked_seq_output = is_first_seg_f * seq_output

        hidden1 = dense(config.hidden_size, "hidden1")(masked_seq_output)
        hidden2 = dense(config.hidden_size, "hidden2")(hidden1)
        self.per_window_output = hidden2

        combined = tf.reduce_max(hidden2, axis=1)
        self.sequence_output = seq_output
        dense_layer = tf.keras.layers.Dense(config.hidden_size,
                                            name="hidden3",
                                            activation=tf.keras.activations.tanh,
                                            kernel_initializer=create_initializer(config.initializer_range))
        hidden3 = dense_layer(combined) # [batch, seq_length, hidden_szie]

        hidden3 = is_first_seg_f[:, 0, :, :] * hidden3
        self.pooled_output = tf.reduce_sum(hidden3, axis=1)


class MultiEvidenceUseFirst(BertModelInterface):
    def __init__(self,
                 config,
                 is_training,
                 input_ids,
                 input_mask=None,
                 token_type_ids=None,
                 use_one_hot_embeddings=True,
                 features=None,
                 scope=None):
        super(MultiEvidenceUseFirst, self).__init__()

        def dense(hidden_size, name):
            return tf.keras.layers.Dense(hidden_size,
                                         activation=tf.keras.activations.tanh,
                                         name=name,
                                         kernel_initializer=create_initializer(config.initializer_range))

        model = BertModel(
                config=config,
                is_training=is_training,
                input_ids=input_ids,
                input_mask=input_mask,
                token_type_ids=token_type_ids,
                use_one_hot_embeddings=use_one_hot_embeddings,
            )
        seq_output = model.get_sequence_output()
        is_first_seg = tf.logical_and(tf.equal(token_type_ids, 0), tf.equal(input_mask, 1))
        is_first_seg_f = tf.expand_dims(tf.cast(is_first_seg, tf.float32), -1)
        masked_seq_output = is_first_seg_f * seq_output
        hidden1 = dense(config.hidden_size, "hidden1")(masked_seq_output)
        hidden2 = dense(config.hidden_size, "hidden2")(hidden1)
        max_seg1_length = 30
        self.predict_output = hidden2[:, :max_seg1_length]
        self.sequence_output = model.get_sequence_output()
        self.pooled_output = model.get_pooled_output()

    def get_output(self):
        return self.predict_output


class MultiEvidenceCombiner:
    def __init__(self,
                 config,
                 is_training,
                 vectors,
                 valid_mask=None,
                 scope=None):
        def dense(hidden_size, name):
            return tf.keras.layers.Dense(hidden_size,
                                         activation=tf.keras.activations.tanh,
                                         name=name,
                                         kernel_initializer=create_initializer(config.initializer_range))

        vectors = tf.cast(tf.expand_dims(valid_mask, -1), tf.float32) * vectors
        #vectors = dense(config.hidden_size, "hidden2")(vectors) # [batch, seq_length, hidden_szie]

        combined = tf.reduce_max(vectors, axis=1)
        dense_layer = tf.keras.layers.Dense(config.hidden_size,
                                            name="hidden3",
                                            activation=tf.keras.activations.tanh,
                                            kernel_initializer=create_initializer(config.initializer_range))
        hidden3 = dense_layer(combined) # [batch, seq_length, hidden_szie]
        #hidden3 = combined
        self.pooled_output = tf.reduce_sum(hidden3, axis=1)