import tensorflow as tf

from tlm.model.base import BertModelInterface, BertModel, create_initializer, get_shape_list2, dropout
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
            # [Batch, num_window, window_length, hidden_size]
            return r3to4(sequence)

        stacked_input_ids, stacked_input_mask, stacked_segment_ids = split_input(input_ids2,
                                                                                 input_mask2,
                                                                                 segment_ids2,
                                                                                 d_seq_length,
                                                                                 unit_length)

        # [Batch, num_window, window_length, hidden_size]
        seq_output = get_seq_output_3d(BertModel, input_ids2, input_mask2, segment_ids2)

        is_first_seg = tf.logical_and(tf.equal(stacked_segment_ids, 0), tf.equal(stacked_input_mask, 1))
        is_first_seg_f = tf.expand_dims(tf.cast(is_first_seg, tf.float32), -1)

        # only use output if the first segment ( perspective sentence )
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


class MESecondProb:
    def __init__(self,
                 config,
                 is_training,
                 use_one_hot_embeddings=True,
                 features=None,
                 scope=None):
        super(MESecondProb, self).__init__()
        self.config = config
        input_ids = features["input_ids"]

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
            # [Batch, num_window, window_length, hidden_size]
            return r3to4(sequence)

        stacked_input_ids, stacked_input_mask, stacked_segment_ids = split_input(input_ids2,
                                                                                 input_mask2,
                                                                                 segment_ids2,
                                                                                 d_seq_length,
                                                                                 unit_length)

        # [Batch, num_window, window_length, hidden_size]
        seq_output = get_seq_output_3d(BertModel, input_ids2, input_mask2, segment_ids2)

        first_token = seq_output[:, :, 0, :]
        hidden1 = dense(config.hidden_size, "hidden1")(first_token)
        pooled = dense(config.hidden_size, "hidden2")(hidden1)
        logits = tf.keras.layers.Dense(2, name="cls_dense")(pooled)
        all_probs = tf.nn.softmax(logits)[:, :, 1]
        self.prob = tf.reduce_mean(all_probs, axis=1)

    def get_prob(self):
        return self.prob

    def get_loss(self, label_ids):
        if self.config.loss == 'hinge':
            return tf.keras.losses.Hinge()(label_ids, self.get_prob())
        elif self.config.loss == 'mae':
            return tf.keras.losses.MAE(label_ids, self.get_prob())
        elif self.config.loss == 'ce':
            return tf.keras.losses.BinaryCrossentropy()(label_ids, self.get_prob())
        else:
            assert False


class MELogit:
    def __init__(self,
                 config,
                 is_training,
                 use_one_hot_embeddings=True,
                 features=None,
                 scope=None):
        super(MELogit, self).__init__()
        self.config = config
        input_ids = features["input_ids"]

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
            # [Batch, num_window, window_length, hidden_size]
            return r3to4(sequence)

        stacked_input_ids, stacked_input_mask, stacked_segment_ids = split_input(input_ids2,
                                                                                 input_mask2,
                                                                                 segment_ids2,
                                                                                 d_seq_length,
                                                                                 unit_length)

        # [Batch, num_window, window_length, hidden_size]
        seq_output = get_seq_output_3d(BertModel, input_ids2, input_mask2, segment_ids2)

        first_token = seq_output[:, :, 0, :]
        hidden1 = dense(config.hidden_size, "hidden1")(first_token)
        pooled = dense(config.hidden_size, "hidden2")(hidden1)
        logits = tf.keras.layers.Dense(2, name="cls_dense")(pooled)
        self.logits = tf.reduce_mean(logits, 1)

    def get_logits(self):
        return self.logits

    def get_loss(self, label_ids):
        y_true = tf.one_hot(label_ids, 2)
        if self.config.loss == 'hinge':
            logits = self.get_logits()
            return tf.keras.losses.Hinge()(label_ids, logits[:, 1])
        elif self.config.loss == 'mae':
            return tf.keras.losses.MAE(y_true, self.get_logits())
        elif self.config.loss == 'ce':
            return tf.keras.losses.BinaryCrossentropy()(y_true, self.get_logits())
        else:
            assert False

# ME : Multi Evidence


class ME3:
    def __init__(self,
                 config,
                 is_training,
                 use_one_hot_embeddings=True,
                 features=None,
                 scope=None):
        super(ME3, self).__init__()
        input_ids = features["input_ids"]

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
            # [Batch, num_window, window_length, hidden_size]
            return r3to4(sequence)

        stacked_input_ids, stacked_input_mask, stacked_segment_ids = split_input(input_ids2,
                                                                                 input_mask2,
                                                                                 segment_ids2,
                                                                                 d_seq_length,
                                                                                 unit_length)

        # [Batch, num_window, window_length, hidden_size]
        seq_output = get_seq_output_3d(BertModel, input_ids2, input_mask2, segment_ids2)

        first_token = seq_output[:, :, 0, :]
        hidden1 = dense(config.hidden_size, "hidden1")(first_token)
        pooled = dense(config.hidden_size, "hidden2")(hidden1)
        logits = tf.keras.layers.Dense(2, name="cls_dense")(pooled)
        all_probs = tf.nn.softmax(logits)[:, :, 1]  # [batch_size, num_window]

        pooled2 = dense(config.hidden_size, "hidden3")(hidden1)
        confidence = tf.keras.layers.Dense(1, name="confidence")(pooled2)
        confidence = tf.reduce_mean(confidence, 2)

        weight = tf.nn.softmax(confidence, axis=1)  # [batch_size, num_window, 1]
        if is_training:
            weight = dropout(weight, 0.5)

        prob = tf.matmul(all_probs, weight, transpose_b=True)
        self.prob = tf.reduce_mean(prob, axis=1)

    def get_prob(self):
        return self.prob


class ME4:
    def __init__(self,
                 config,
                 is_training,
                 use_one_hot_embeddings=True,
                 features=None,
                 scope=None):
        super(ME4, self).__init__()
        input_ids = features["input_ids"]

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
            # [Batch, num_window, window_length, hidden_size]
            return r3to4(sequence)

        stacked_input_ids, stacked_input_mask, stacked_segment_ids = split_input(input_ids2,
                                                                                 input_mask2,
                                                                                 segment_ids2,
                                                                                 d_seq_length,
                                                                                 unit_length)

        # [Batch, num_window, window_length, hidden_size]
        seq_output = get_seq_output_3d(BertModel, input_ids2, input_mask2, segment_ids2)

        first_token = seq_output[:, :, 0, :]
        hidden1 = dense(config.hidden_size, "hidden1")(first_token)
        pooled = dense(config.hidden_size, "hidden2")(hidden1)
        logits = tf.keras.layers.Dense(2, name="cls_dense")(pooled)
        probs = tf.nn.softmax(logits)[:, :, 1]  # [batch_size, num_window]
        weight = tf.nn.softmax(probs, axis=1)  # [batch_size, num_window]
        if is_training:
            weight = dropout(weight, 0.1)
        prob = tf.matmul(probs, weight, transpose_b=True)
        self.prob = tf.reduce_mean(prob, axis=1) # just remove the last dimension

    def get_prob(self):
        return self.prob


class ME4_2(BertModelInterface):
    def __init__(self,
                 config,
                 is_training,
                 input_ids,
                 input_mask=None,
                 token_type_ids=None,
                 use_one_hot_embeddings=True,
                 features=None,
                 scope=None):
        super(ME4_2, self).__init__()
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
            # [Batch, num_window, window_length, hidden_size]
            return r3to4(sequence)

        stacked_input_ids, stacked_input_mask, stacked_segment_ids = split_input(input_ids2,
                                                                                 input_mask2,
                                                                                 segment_ids2,
                                                                                 d_seq_length,
                                                                                 unit_length)

        with tf.compat.v1.variable_scope(dual_model_prefix1):
            model_1 = BertModel(
                config=config,
                is_training=is_training,
                input_ids=input_ids,
                input_mask=input_mask,
                token_type_ids=token_type_ids,
                use_one_hot_embeddings=use_one_hot_embeddings,
            )

        # [Batch, num_window, window_length, hidden_size]
        with tf.compat.v1.variable_scope(dual_model_prefix2):
            print(input_ids2)
            seq_output = get_seq_output_3d(BertModel, input_ids2, input_mask2, segment_ids2)
            print(seq_output)

            first_token = seq_output[:, :, 0, :]
            hidden1 = dense(config.hidden_size, "hidden1")(first_token)
            pooled = dense(config.hidden_size, "hidden2")(hidden1)
            logits = tf.keras.layers.Dense(2, name="cls_dense")(pooled)
            probs = tf.nn.softmax(logits)[:, :, 1]  # [batch_size, num_window]
            weight = tf.nn.softmax(probs, axis=1)  # [batch_size, num_window]
            if is_training:
                weight = dropout(weight, 0.1)

            prob = tf.matmul(probs, weight, transpose_b=True)

        model_1_first_token = model_1.get_sequence_output()[:, 0, :]
        rep = tf.concat([model_1_first_token, prob], axis=1)
        dense_layer = tf.keras.layers.Dense(config.hidden_size,
                                            activation=tf.keras.activations.tanh,
                                            kernel_initializer=create_initializer(config.initializer_range))
        pooled_output = dense_layer(rep)
        self.pooled_output = pooled_output


class ME5:
    def __init__(self,
                 config,
                 is_training,
                 use_one_hot_embeddings=True,
                 features=None,
                 scope=None):
        super(ME5, self).__init__()
        input_ids = features["input_ids"]

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
            # [Batch, num_window, window_length, hidden_size]
            return r3to4(sequence)

        stacked_input_ids, stacked_input_mask, stacked_segment_ids = split_input(input_ids2,
                                                                                 input_mask2,
                                                                                 segment_ids2,
                                                                                 d_seq_length,
                                                                                 unit_length)

        # [Batch, num_window, window_length, hidden_size]
        seq_output = get_seq_output_3d(BertModel, input_ids2, input_mask2, segment_ids2)

        first_token = seq_output[:, :, 0, :]
        hidden1 = dense(config.hidden_size, "hidden1")(first_token)
        pooled = dense(config.hidden_size, "hidden2")(hidden1)
        logits = tf.keras.layers.Dense(2, name="cls_dense")(pooled)
        probs = tf.nn.softmax(logits)[:, :, 1]  # [batch_size, num_window]
        probs_sorted = tf.sort(probs, axis=1, direction='DESCENDING')
        self.probs_sorted = probs_sorted
        if is_training:
            probs_sorted = dropout(probs_sorted, 0.1)

        h1 = dense(num_window, "weight_hidden")(probs_sorted)
        weights_logits = dense(num_window, "weight_hidden2")(h1)
        weight = tf.nn.softmax(weights_logits, axis=1)  # [batch_size, num_window]

        self.weight = weight
        prob = tf.matmul(probs_sorted, weight, transpose_b=True)
        self.prob = tf.reduce_mean(prob, axis=1) # just remove the last dimension

    def get_prob(self):
        return self.prob


class ME5_2(BertModelInterface):
    def __init__(self,
                 config,
                 is_training,
                 input_ids,
                 input_mask=None,
                 token_type_ids=None,
                 use_one_hot_embeddings=True,
                 features=None,
                 scope=None):
        super(ME5_2, self).__init__()
        input_ids = features["input_ids"]

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
            # [Batch, num_window, window_length, hidden_size]
            return r3to4(sequence)

        stacked_input_ids, stacked_input_mask, stacked_segment_ids = split_input(input_ids2,
                                                                                 input_mask2,
                                                                                 segment_ids2,
                                                                                 d_seq_length,
                                                                                 unit_length)

        # [Batch, num_window, window_length, hidden_size]
        seq_output = get_seq_output_3d(BertModel, input_ids2, input_mask2, segment_ids2)

        first_token = seq_output[:, :, 0, :]
        hidden1 = dense(config.hidden_size, "hidden1")(first_token)
        pooled = dense(config.hidden_size, "hidden2")(hidden1)
        logits = tf.keras.layers.Dense(2, name="cls_dense")(pooled)
        probs = tf.nn.softmax(logits)[:, :, 1]  # [batch_size, num_window]
        probs_sorted = probs
        # probs_sorted = tf.sort(probs, axis=1, direction='DESCENDING')
        if is_training:
            probs_sorted = dropout(probs_sorted, 0.1)

        pad_len = config.hidden_size - num_window
        self.pooled_output = tf.concat([probs_sorted, tf.zeros([batch_size, pad_len])], axis=1)
        print(self.pooled_output)



class ME6:
    def __init__(self,
                 config,
                 is_training,
                 use_one_hot_embeddings=True,
                 features=None,
                 scope=None):
        super(ME6, self).__init__()
        input_ids = features["input_ids"]

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
            # [Batch, num_window, window_length, hidden_size]
            return r3to4(sequence)

        stacked_input_ids, stacked_input_mask, stacked_segment_ids = split_input(input_ids2,
                                                                                 input_mask2,
                                                                                 segment_ids2,
                                                                                 d_seq_length,
                                                                                 unit_length)

        # [Batch, num_window, window_length, hidden_size]
        seq_output = get_seq_output_3d(BertModel, input_ids2, input_mask2, segment_ids2)

        first_token = seq_output[:, :, 0, :]
        hidden1 = dense(config.hidden_size, "hidden1")(first_token)
        pooled = dense(config.hidden_size, "hidden2")(hidden1)
        logits = tf.keras.layers.Dense(2, name="cls_dense")(pooled)
        probs = tf.nn.softmax(logits)[:, :, 1]  # [batch_size, num_window]
        probs_sorted = tf.sort(probs, axis=1, direction='DESCENDING')
        self.probs_sorted = probs_sorted
        if is_training:
            probs_sorted = dropout(probs_sorted, 0.1)

        weights_logits = tf.compat.v1.get_variable(
            "weights_logits", [1, num_window],
            initializer=tf.compat.v1.zeros_initializer()
        )
        weight = tf.nn.softmax(weights_logits, axis=1)  # [batch_size, num_window]

        self.weight = weight
        prob = tf.matmul(probs_sorted, weight, transpose_b=True)
        self.prob = tf.reduce_mean(prob, axis=1) # just remove the last dimension

    def get_prob(self):
        return self.prob


class ME7(BertModelInterface):
    def __init__(self,
                 config,
                 is_training,
                 input_ids,
                 input_mask=None,
                 token_type_ids=None,
                 use_one_hot_embeddings=True,
                 features=None,
                 scope=None):
        super(ME7, self).__init__()
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
            # [Batch, num_window, window_length, hidden_size]
            return r3to4(sequence)

        stacked_input_ids, stacked_input_mask, stacked_segment_ids = split_input(input_ids2,
                                                                                 input_mask2,
                                                                                 segment_ids2,
                                                                                 d_seq_length,
                                                                                 unit_length)

        with tf.compat.v1.variable_scope(dual_model_prefix1):
            model_1 = BertModel(
                config=config,
                is_training=is_training,
                input_ids=input_ids,
                input_mask=input_mask,
                token_type_ids=token_type_ids,
                use_one_hot_embeddings=use_one_hot_embeddings,
            )

        # [Batch, num_window, window_length, hidden_size]
        with tf.compat.v1.variable_scope(dual_model_prefix2):
            print(input_ids2)
            seq_output = get_seq_output_3d(BertModel, input_ids2, input_mask2, segment_ids2)
            print(seq_output)

            model_2_first_token = seq_output[:, :, 0, :] # batch, num_window, hidden_size

        model_1_first_token = model_1.get_sequence_output()[:, :1, :] # batch, 1, hidden_size]
        m1ft = tf.tile(model_1_first_token, [1, num_window, 1])
        print(m1ft)

        common_rep = tf.concat([m1ft, model_2_first_token], axis=2) # batch, num_window, hidden_size]
        dense_layer1 = tf.keras.layers.Dense(config.hidden_size * 4,
                                            activation=tf.keras.activations.tanh,
                                            kernel_initializer=create_initializer(config.initializer_range))
        dense_layer2 = tf.keras.layers.Dense(config.hidden_size,
                                             activation=tf.keras.activations.tanh,
                                             kernel_initializer=create_initializer(config.initializer_range))
        hidden1 = dense_layer1(common_rep)
        hidden2 = dense_layer2(hidden1)
        self.pooled_output = tf.reduce_max(hidden2, axis=1)


class MEDirect(BertModelInterface):
    def __init__(self,
                 config,
                 is_training,
                 use_one_hot_embeddings=True,
                 features=None,
                 scope=None):
        super(MEDirect, self).__init__()
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]

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
                token_type_ids=segment_ids,
                use_one_hot_embeddings=use_one_hot_embeddings,
            )

        with tf.compat.v1.variable_scope(triple_model_prefix2):
            pooled2 = get_pooled_output_3d(BertModel, input_ids2, input_mask2, segment_ids2)

        bert1_rep = model_1.get_sequence_output()[:, 0, :]
        bert1_rep = tf.expand_dims(bert1_rep, 1)
        bert1_rep = tf.tile(bert1_rep, [1, num_window, 1]) #
        combined_rep = tf.concat([bert1_rep, pooled2], axis=2)
        hidden1 = dense(config.hidden_size, "hidden1")(combined_rep)
        hidden2 = dense(config.hidden_size, "hidden2")(hidden1)
        dense_layer = tf.keras.layers.Dense(2,
                                            name="hidden3",
                                            activation=tf.keras.activations.tanh,
                                            kernel_initializer=create_initializer(config.initializer_range))
        logits = dense_layer(hidden2)
        label_ids = features["label_ids"]
        all_probs = tf.nn.softmax(logits)
        self.prob = tf.reduce_mean(all_probs, axis=1)
        epsilon = 1e-6
        probs_ad = tf.clip_by_value(self.prob, epsilon, 1.0 - epsilon)
        y_true = tf.one_hot(label_ids, 2)
        loss_arr = tf.keras.losses.BinaryCrossentropy()(y_true, probs_ad)
        loss = tf.reduce_mean(loss_arr)
        self.loss = loss

    def get_prob(self):
        return self.prob

    def get_logits(self):
        epsilon = 1e-6
        probs_ad = tf.clip_by_value(self.prob, epsilon, 1.0 - epsilon)
        return tf.math.log(probs_ad)

    def get_loss(self, label_ids):
        return self.loss
