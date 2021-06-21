
import tensorflow as tf

# Final probability is mean of each segments
from models.transformer.bert_common_v2 import get_shape_list2
from tlm.model.base import BertModelInterface, BertModel
from tlm.model.dual_model_common import dual_model_prefix1, dual_model_prefix2
from tlm.model.multiple_evidence import split_input, r3to2


class MES_lower_only(BertModelInterface):
    def __init__(self,
                 config,
                 is_training,
                 use_one_hot_embeddings=True,
                 features=None,
                 scope=None):

        super(MES_lower_only, self).__init__()
        alpha = config.alpha
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]

        unit_length = config.max_seq_length
        d_seq_length = config.max_d_seq_length
        num_window = int(d_seq_length / unit_length)
        batch_size, _ = get_shape_list2(input_ids)

        # [Batch, num_window, unit_seq_length]
        stacked_input_ids, stacked_input_mask, stacked_segment_ids = split_input(input_ids,
                                                                                 input_mask,
                                                                                 segment_ids,
                                                                                 d_seq_length,
                                                                                 unit_length)
        # Ignore the window if
        # 1. The window is not first window and
        #   1.1 All input_mask is 0
        #   1.2 Content is too short, number of document tokens (other than query tokens) < 10

        # [Batch, num_window]
        is_first_window = tf.concat([tf.ones([batch_size, 1], tf.bool),
                                     tf.zeros([batch_size, num_window-1], tf.bool)], axis=1)
        num_content_tokens = tf.reduce_sum(stacked_segment_ids, 2)
        has_enough_evidence = tf.less(10, num_content_tokens)
        is_valid_window = tf.logical_or(is_first_window, has_enough_evidence)
        is_valid_window_mask = tf.cast(is_valid_window, tf.float32)
        self.is_first_window = is_first_window
        self.num_content_tokens = num_content_tokens
        self.has_enough_evidence = has_enough_evidence
        self.is_valid_window = is_valid_window
        self.is_valid_window_mask = is_valid_window_mask
        self.has_any_content = tf.less(1, num_content_tokens)


        with tf.compat.v1.variable_scope(dual_model_prefix1):
            model = BertModel(
                config=config,
                is_training=is_training,
                input_ids=r3to2(stacked_input_ids),
                input_mask=r3to2(stacked_input_mask),
                token_type_ids=r3to2(stacked_segment_ids),
                use_one_hot_embeddings=use_one_hot_embeddings,
            )
            pooled = model.get_pooled_output()
            logits_2d = tf.keras.layers.Dense(2, name="cls_dense")(pooled)  #

        def r2to3(arr):
            return tf.reshape(arr, [batch_size, num_window, -1])


        # [Batch, num_window, window_length, hidden_size]
        logits_3d = r2to3(logits_2d)
        probs = tf.nn.softmax(logits_3d)[:, :, 1]  # [batch_size, num_window]
        valid_probs = probs * is_valid_window_mask
        max_seg = tf.argmax(valid_probs, axis=1)
        self.max_seg = max_seg

        label_ids_repeat = tf.tile(label_ids, [1, num_window])
        loss_arr = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits_3d,
            labels=label_ids_repeat)
        loss_arr = loss_arr * is_valid_window_mask
        layer1_loss = tf.reduce_mean(loss_arr)
        self.logits_3d = logits_3d
        self.loss = layer1_loss
        self.logits = logits_2d

    def get_loss(self, label_ids_not_used):
        return self.loss

    def get_logits(self):
        return self.logits


class MES_continuous_0(BertModelInterface):
    def __init__(self,
                 config,
                 is_training,
                 use_one_hot_embeddings=True,
                 features=None,
                 scope=None):

        super(MES_continuous_0, self).__init__()
        alpha = config.alpha
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]

        unit_length = config.max_seq_length
        d_seq_length = config.max_d_seq_length
        num_window = int(d_seq_length / unit_length)
        batch_size, _ = get_shape_list2(input_ids)

        # [Batch, num_window, unit_seq_length]
        stacked_input_ids, stacked_input_mask, stacked_segment_ids = split_input(input_ids,
                                                                                 input_mask,
                                                                                 segment_ids,
                                                                                 d_seq_length,
                                                                                 unit_length)
        # Ignore the window if
        # 1. The window is not first window and
        #   1.1 All input_mask is 0
        #   1.2 Content is too short, number of document tokens (other than query tokens) < 10

        # [Batch, num_window]
        is_first_window = tf.concat([tf.ones([batch_size, 1], tf.bool),
                                     tf.zeros([batch_size, num_window-1], tf.bool)], axis=1)
        num_content_tokens = tf.reduce_sum(stacked_segment_ids, 2)
        has_enough_evidence = tf.less(10, num_content_tokens)
        is_valid_window = tf.logical_or(is_first_window, has_enough_evidence)
        is_valid_window_mask = tf.cast(is_valid_window, tf.float32)
        self.is_first_window = is_first_window
        self.num_content_tokens = num_content_tokens
        self.has_enough_evidence = has_enough_evidence
        self.is_valid_window = is_valid_window
        self.is_valid_window_mask = is_valid_window_mask

        with tf.compat.v1.variable_scope(dual_model_prefix1):
            model = BertModel(
                config=config,
                is_training=is_training,
                input_ids=r3to2(stacked_input_ids),
                input_mask=r3to2(stacked_input_mask),
                token_type_ids=r3to2(stacked_segment_ids),
                use_one_hot_embeddings=use_one_hot_embeddings,
            )
            pooled = model.get_pooled_output()
            logits_2d = tf.keras.layers.Dense(2, name="cls_dense")(pooled)  #

        def r2to3(arr):
            return tf.reshape(arr, [batch_size, num_window, -1])

        # [Batch, num_window, window_length, hidden_size]
        logits_3d = r2to3(logits_2d)
        label_ids_repeat = tf.tile(label_ids, [1, num_window])
        loss_arr = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits_3d,
            labels=label_ids_repeat)
        loss_arr = loss_arr * is_valid_window_mask
        layer1_loss = tf.reduce_mean(loss_arr)

        probs = tf.nn.softmax(logits_3d)[:, :, 1]  # [batch_size, num_window]

        # Probabilistic selection
        def select_seg(stacked_input_ids, indices):
            # indices : [batch_size, 1]
            return tf.gather(stacked_input_ids, indices, axis=1, batch_dims=1)

        valid_probs = probs * is_valid_window_mask
        max_seg = tf.argmax(valid_probs, axis=1)
        input_ids = select_seg(stacked_input_ids, max_seg)
        input_mask = select_seg(stacked_input_mask, max_seg)
        segment_ids = select_seg(stacked_segment_ids, max_seg)

        with tf.compat.v1.variable_scope(dual_model_prefix2):
            model = BertModel(
                config=config,
                is_training=is_training,
                input_ids=input_ids,
                input_mask=input_mask,
                token_type_ids=segment_ids,
                use_one_hot_embeddings=use_one_hot_embeddings,
            )
            logits = tf.keras.layers.Dense(2, name="cls_dense")(model.get_pooled_output())
            self.logits = logits
        label_ids = tf.reshape(label_ids, [-1])
        loss_arr = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits,
            labels=label_ids)

        layer2_loss = tf.reduce_mean(loss_arr)
        loss = alpha * layer1_loss + layer2_loss
        self.loss = loss

    def get_loss(self, label_ids_not_used):
        return self.loss

    def get_logits(self):
        return self.logits


class MES_continuous_0_nce(BertModelInterface):
    def __init__(self,
                 config,
                 is_training,
                 use_one_hot_embeddings=True,
                 features=None,
                 scope=None):

        super(MES_continuous_0_nce, self).__init__()
        alpha = config.alpha
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]

        unit_length = config.max_seq_length
        d_seq_length = config.max_d_seq_length
        num_window = int(d_seq_length / unit_length)
        batch_size, _ = get_shape_list2(input_ids)

        # [Batch, num_window, unit_seq_length]
        stacked_input_ids, stacked_input_mask, stacked_segment_ids = split_input(input_ids,
                                                                                 input_mask,
                                                                                 segment_ids,
                                                                                 d_seq_length,
                                                                                 unit_length)
        # Ignore the window if
        # 1. The window is not first window and
        #   1.1 All input_mask is 0
        #   1.2 Content is too short, number of document tokens (other than query tokens) < 10

        # [Batch, num_window]
        is_first_window = tf.concat([tf.ones([batch_size, 1], tf.bool),
                                     tf.zeros([batch_size, num_window-1], tf.bool)], axis=1)
        num_content_tokens = tf.reduce_sum(stacked_segment_ids, 2)
        has_enough_evidence = tf.less(10, num_content_tokens)
        is_valid_window = tf.logical_or(is_first_window, has_enough_evidence)
        is_valid_window_mask = tf.cast(is_valid_window, tf.float32)
        self.is_first_window = is_first_window
        self.num_content_tokens = num_content_tokens
        self.has_enough_evidence = has_enough_evidence
        self.is_valid_window = is_valid_window
        self.is_valid_window_mask = is_valid_window_mask

        with tf.compat.v1.variable_scope(dual_model_prefix1):
            model = BertModel(
                config=config,
                is_training=is_training,
                input_ids=r3to2(stacked_input_ids),
                input_mask=r3to2(stacked_input_mask),
                token_type_ids=r3to2(stacked_segment_ids),
                use_one_hot_embeddings=use_one_hot_embeddings,
            )
            pooled = model.get_pooled_output()
            logits_2d = tf.keras.layers.Dense(2, name="cls_dense")(pooled)  #

        def r2to3(arr):
            return tf.reshape(arr, [batch_size, num_window, -1])

        # [Batch, num_window, window_length, hidden_size]
        logits_3d = r2to3(logits_2d)
        label_ids_repeat = tf.tile(label_ids, [1, num_window])
        loss_arr = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits_3d,
            labels=label_ids_repeat)

        c = -tf.math.log(0.7)
        loss_arr = tf.maximum(loss_arr, c)

        loss_arr = loss_arr * is_valid_window_mask
        layer1_loss = tf.reduce_mean(loss_arr)

        probs = tf.nn.softmax(logits_3d)[:, :, 1]  # [batch_size, num_window]

        # Probabilistic selection
        def select_seg(stacked_input_ids, indices):
            # indices : [batch_size, 1]
            return tf.gather(stacked_input_ids, indices, axis=1, batch_dims=1)

        valid_probs = probs * is_valid_window_mask
        max_seg = tf.argmax(valid_probs, axis=1)
        input_ids = select_seg(stacked_input_ids, max_seg)
        input_mask = select_seg(stacked_input_mask, max_seg)
        segment_ids = select_seg(stacked_segment_ids, max_seg)

        with tf.compat.v1.variable_scope(dual_model_prefix2):
            model = BertModel(
                config=config,
                is_training=is_training,
                input_ids=input_ids,
                input_mask=input_mask,
                token_type_ids=segment_ids,
                use_one_hot_embeddings=use_one_hot_embeddings,
            )
            logits = tf.keras.layers.Dense(2, name="cls_dense")(model.get_pooled_output())
            self.logits = logits
        label_ids = tf.reshape(label_ids, [-1])
        loss_arr = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits,
            labels=label_ids)

        layer2_loss = tf.reduce_mean(loss_arr)
        loss = alpha * layer1_loss + layer2_loss
        self.loss = loss

    def get_loss(self, label_ids_not_used):

        return self.loss

    def get_logits(self):
        return self.logits


class MES_continuous_1(BertModelInterface):
    def __init__(self,
                 config,
                 is_training,
                 use_one_hot_embeddings=True,
                 features=None,
                 scope=None):

        super(MES_continuous_1, self).__init__()
        alpha = config.alpha
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]

        unit_length = config.max_seq_length
        d_seq_length = config.max_d_seq_length
        num_window = int(d_seq_length / unit_length)
        batch_size, _ = get_shape_list2(input_ids)

        # [Batch, num_window, unit_seq_length]
        stacked_input_ids, stacked_input_mask, stacked_segment_ids = split_input(input_ids,
                                                                                 input_mask,
                                                                                 segment_ids,
                                                                                 d_seq_length,
                                                                                 unit_length)
        # Ignore the window if
        # 1. The window is not first window and
        #   1.1 All input_mask is 0
        #   1.2 Content is too short, number of document tokens (other than query tokens) < 10

        # [Batch, num_window]
        is_first_window = tf.concat([tf.ones([batch_size, 1], tf.bool),
                                     tf.zeros([batch_size, num_window-1], tf.bool)], axis=1)
        num_content_tokens = tf.reduce_sum(stacked_segment_ids, 2)
        has_enough_evidence = tf.less(10, num_content_tokens)
        is_valid_window = tf.logical_or(is_first_window, has_enough_evidence)
        is_valid_window_mask = tf.cast(is_valid_window, tf.float32)
        self.is_first_window = is_first_window
        self.num_content_tokens = num_content_tokens
        self.has_enough_evidence = has_enough_evidence
        self.is_valid_window = is_valid_window
        self.is_valid_window_mask = is_valid_window_mask

        def r2to3(arr):
            return tf.reshape(arr, [batch_size, num_window, -1])

        with tf.compat.v1.variable_scope(dual_model_prefix1):
            model = BertModel(
                config=config,
                is_training=is_training,
                input_ids=r3to2(stacked_input_ids),
                input_mask=r3to2(stacked_input_mask),
                token_type_ids=r3to2(stacked_segment_ids),
                use_one_hot_embeddings=use_one_hot_embeddings,
            )

            # [Batch, num_window, window_length, hidden_size]
            pooled = model.get_pooled_output()
            logits_2d = tf.keras.layers.Dense(2, name="cls_dense")(pooled) #

        logits_3d = r2to3(logits_2d)

        probs = tf.nn.softmax(logits_3d)[:, :, 1]  # [batch_size, num_window]

        # Probabilistic selection
        def select_seg(stacked_input_ids, indices):
            # indices : [batch_size, 1]
            return tf.gather(stacked_input_ids, indices, axis=1, batch_dims=1)

        valid_probs = probs * is_valid_window_mask
        max_seg = tf.argmax(valid_probs, axis=1)
        input_ids = select_seg(stacked_input_ids, max_seg)
        input_mask = select_seg(stacked_input_mask, max_seg)
        segment_ids = select_seg(stacked_segment_ids, max_seg)

        with tf.compat.v1.variable_scope(dual_model_prefix2):
            model = BertModel(
                config=config,
                is_training=is_training,
                input_ids=input_ids,
                input_mask=input_mask,
                token_type_ids=segment_ids,
                use_one_hot_embeddings=use_one_hot_embeddings,
            )
            logits = tf.keras.layers.Dense(2, name="cls_dense")(model.get_pooled_output())
        self.logits = logits
        label_ids = tf.reshape(label_ids, [-1])
        loss_arr = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits,
            labels=label_ids)

        layer2_loss = tf.reduce_mean(loss_arr)
        loss = layer2_loss
        self.loss = loss

    def get_loss(self, label_ids_not_used):
        return self.loss

    def get_logits(self):
        return self.logits



# Segment wise
class MES_cont_pred(BertModelInterface):
    def __init__(self,
                 config,
                 is_training,
                 use_one_hot_embeddings=True,
                 features=None,
                 scope=None):

        super(MES_cont_pred, self).__init__()
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]

        unit_length = config.max_seq_length
        d_seq_length = config.max_d_seq_length
        batch_size, _ = get_shape_list2(input_ids)

        # [Batch, unit_seq_length]
        with tf.compat.v1.variable_scope(dual_model_prefix1):
            model = BertModel(
                config=config,
                is_training=is_training,
                input_ids=input_ids,
                input_mask=input_mask,
                token_type_ids=segment_ids,
                use_one_hot_embeddings=use_one_hot_embeddings,
            )
            pooled = model.get_pooled_output()
            logits_2d = tf.keras.layers.Dense(2, name="cls_dense")(pooled) #


        with tf.compat.v1.variable_scope(dual_model_prefix2):
            model = BertModel(
                config=config,
                is_training=is_training,
                input_ids=input_ids,
                input_mask=input_mask,
                token_type_ids=segment_ids,
                use_one_hot_embeddings=use_one_hot_embeddings,
            )
            logits = tf.keras.layers.Dense(2, name="cls_dense")(model.get_pooled_output())
        self.logits = logits
        label_ids = tf.reshape(label_ids, [-1])
        loss_arr = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits,
            labels=label_ids)

        layer2_loss = tf.reduce_mean(loss_arr)
        self.loss = layer2_loss

    def get_loss(self, label_ids_not_used):
        return self.loss

    def get_logits(self):
        return self.logits
