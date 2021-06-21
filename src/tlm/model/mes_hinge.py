import tensorflow as tf

# Final probability is mean of each segments
from models.transformer.bert_common_v2 import get_shape_list2
from tlm.model.base import BertModelInterface, BertModel
from tlm.model.dual_model_common import dual_model_prefix1, dual_model_prefix2
from tlm.model.multiple_evidence import split_input, r3to2


class MES_hinge(BertModelInterface):
    def __init__(self,
                 config,
                 is_training,
                 use_one_hot_embeddings=True,
                 features=None,
                 scope=None):

        super(MES_hinge, self).__init__()
        alpha = config.alpha
        input_ids1 = features["input_ids1"]
        input_mask1 = features["input_mask1"]
        segment_ids1 = features["segment_ids1"]
        input_ids2 = features["input_ids2"]
        input_mask2 = features["input_mask2"]
        segment_ids2 = features["segment_ids2"]

        input_ids = tf.concat([input_ids1, input_ids2], axis=0)
        input_mask = tf.concat([input_mask1, input_mask2], axis=0)
        segment_ids = tf.concat([segment_ids1, segment_ids2], axis=0)

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
        self.has_any_content = tf.less(1, num_content_tokens)
        self.is_valid_window = is_valid_window

        with tf.compat.v1.variable_scope(dual_model_prefix1):
            model = BertModel(
                config=config,
                is_training=is_training,
                input_ids=r3to2(stacked_input_ids),
                input_mask=r3to2(stacked_input_mask),
                token_type_ids=r3to2(stacked_segment_ids),
                use_one_hot_embeddings=use_one_hot_embeddings,
            )

        def r2to3(arr):
            return tf.reshape(arr, [batch_size, num_window, -1])

        # [Batch, num_window, window_length, hidden_size]
        pooled = model.get_pooled_output()
        logits_2d = tf.keras.layers.Dense(1, name="cls_dense")(pooled) #
        logits_3d = r2to3(logits_2d) # [ batch, num_window, 1]
        pair_logits_layer1 = tf.reshape(logits_3d, [2, -1, num_window])
        y_diff = pair_logits_layer1[0, :, :] - pair_logits_layer1[1, :, :]
        loss_arr = tf.maximum(1.0 - y_diff, 0)

        is_valid_window_pair = tf.reshape(is_valid_window, [2, -1, num_window])
        is_valid_window_and = tf.logical_and(is_valid_window_pair[0, :, :],
                                                  is_valid_window_pair[1, :, :])
        is_valid_window_paired_mask = tf.cast(is_valid_window_and, tf.float32)
        loss_arr = loss_arr * is_valid_window_paired_mask
        layer1_loss = tf.reduce_mean(loss_arr)

        # probs = tf.nn.softmax(logits_3d)[:, :, 0]  # [batch_size, num_window]
        wrong_probs = tf.nn.softmax(logits_3d)[:, :, 0]  # [batch_size, num_window]

        raw_layer1_scores = logits_3d[:, :, 0]
        self.raw_layer1_scores = raw_layer1_scores
        self.logits3_d = logits_3d

        # Probabilistic selection
        def select_seg(stacked_input_ids, indices):
            # indices : [batch_size, 1]
            return tf.gather(stacked_input_ids, indices, axis=1, batch_dims=1)

        adder = (1.0 - is_valid_window_mask) * -10000.0

        layer1_scores = raw_layer1_scores + adder
        self.layer1_scores = layer1_scores
        max_seg = tf.argmax(layer1_scores, axis=1)
        self.max_seg = max_seg
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
        logits = tf.keras.layers.Dense(1, name="cls_dense")(model.get_pooled_output())
        pair_logits = tf.reshape(logits, [2, -1])
        y_diff2 = pair_logits[0, :] - pair_logits[1, :]
        loss_arr = tf.maximum(1.0 - y_diff2, 0)
        self.logits = logits
        layer2_loss = tf.reduce_mean(loss_arr)
        loss = alpha * layer1_loss + layer2_loss
        self.loss = loss

    def get_loss(self):
        return self.loss

    def get_logits(self):
        return self.logits


class MES_hinge_layer1_load(BertModelInterface):
    def __init__(self,
                 config,
                 is_training,
                 use_one_hot_embeddings=True,
                 features=None,
                 scope=None):

        super(MES_hinge_layer1_load, self).__init__()
        alpha = config.alpha
        input_ids1 = features["input_ids1"]
        input_mask1 = features["input_mask1"]
        segment_ids1 = features["segment_ids1"]
        input_ids2 = features["input_ids2"]
        input_mask2 = features["input_mask2"]
        segment_ids2 = features["segment_ids2"]

        input_ids = tf.concat([input_ids1, input_ids2], axis=0)
        input_mask = tf.concat([input_mask1, input_mask2], axis=0)
        segment_ids = tf.concat([segment_ids1, segment_ids2], axis=0)

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
            logits_2d = tf.keras.layers.Dense(1, name="cls_dense")(pooled)  #

        def r2to3(arr):
            return tf.reshape(arr, [batch_size, num_window, -1])

        # [Batch, num_window, window_length, hidden_size]
        logits_3d = r2to3(logits_2d) # [ batch, num_window, 1]
        pair_logits_layer1 = tf.reshape(logits_3d, [2, -1, num_window])
        y_diff = pair_logits_layer1[0, :, :] - pair_logits_layer1[1, :, :]

        is_valid_window_pair = tf.reshape(is_valid_window, [2, -1, num_window])
        is_valid_window_and = tf.logical_and(is_valid_window_pair[0, :, :],
                                                  is_valid_window_pair[1, :, :])
        # probs = tf.nn.softmax(logits_3d)[:, :, 0]  # [batch_size, num_window]
        probs = logits_3d[:, :, 0]  # [batch_size, num_window]

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
            logits = tf.keras.layers.Dense(1, name="cls_dense")(model.get_pooled_output())
        pair_logits = tf.reshape(logits, [2, -1])
        y_diff2 = pair_logits[0, :] - pair_logits[1, :]
        loss_arr = tf.maximum(1.0 - y_diff2, 0)
        self.logits = logits
        layer2_loss = tf.reduce_mean(loss_arr)
        self.loss = layer2_loss

    def get_loss(self):
        return self.loss

    def get_logits(self):
        return self.logits


class MES_pred(BertModelInterface):
    def __init__(self,
                 config,
                 is_training,
                 use_one_hot_embeddings=True,
                 features=None,
                 scope=None):

        super(MES_pred, self).__init__()
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

        with tf.compat.v1.variable_scope(dual_model_prefix1):
            model = BertModel(
                config=config,
                is_training=is_training,
                input_ids=r3to2(stacked_input_ids),
                input_mask=r3to2(stacked_input_mask),
                token_type_ids=r3to2(stacked_segment_ids),
                use_one_hot_embeddings=use_one_hot_embeddings,
            )

        def r2to3(arr):
            return tf.reshape(arr, [batch_size, num_window, -1])

        # [Batch, num_window, window_length, hidden_size]
        pooled = model.get_pooled_output()
        logits_2d = tf.keras.layers.Dense(1, name="cls_dense")(pooled) #
        logits_3d = r2to3(logits_2d) # [ batch, num_window, 1]


        probs = tf.nn.softmax(logits_3d)[:, :, 0]  # [batch_size, num_window]

        # Probabilistic selection
        def select_seg(stacked_input_ids, indices):
            # indices : [batch_size, 1]
            return tf.gather(stacked_input_ids, indices, axis=1, batch_dims=1)

        valid_probs = probs * is_valid_window_mask
        with tf.compat.v1.variable_scope(dual_model_prefix2):
            model = BertModel(
                config=config,
                is_training=is_training,
                input_ids=input_ids,
                input_mask=input_mask,
                token_type_ids=segment_ids,
                use_one_hot_embeddings=use_one_hot_embeddings,
            )
        logits = tf.keras.layers.Dense(1, name="cls_dense")(model.get_pooled_output())
        self.logits = logits
        self.loss = tf.constant(0)

    def get_loss(self):
        return self.loss

    def get_logits(self):
        return self.logits


class MES_pred_layer2_load(BertModelInterface):
    def __init__(self,
                 config,
                 is_training,
                 use_one_hot_embeddings=True,
                 features=None,
                 scope=None):

        super(MES_pred_layer2_load, self).__init__()
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]

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

        self.is_first_window = is_first_window
        self.num_content_tokens = num_content_tokens
        self.has_enough_evidence = has_enough_evidence
        self.is_valid_window = is_valid_window

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
            logits_2d = tf.keras.layers.Dense(1, name="cls_dense")(pooled)  #
        with tf.compat.v1.variable_scope(dual_model_prefix2):
            model = BertModel(
                config=config,
                is_training=is_training,
                input_ids=input_ids,
                input_mask=input_mask,
                token_type_ids=segment_ids,
                use_one_hot_embeddings=use_one_hot_embeddings,
            )
            logits = tf.keras.layers.Dense(1, name="cls_dense")(model.get_pooled_output())
        self.logits = logits
        self.loss = tf.constant(0)

    def get_loss(self):
        return self.loss

    def get_logits(self):
        return self.logits


class MES_hinge_pred_with_layer1(BertModelInterface):
    def __init__(self,
                 config,
                 is_training,
                 use_one_hot_embeddings=True,
                 features=None,
                 scope=None):

        super(MES_hinge_pred_with_layer1, self).__init__()
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


        with tf.compat.v1.variable_scope(dual_model_prefix1):
            model = BertModel(
                config=config,
                is_training=is_training,
                input_ids=input_ids,
                input_mask=input_mask,
                token_type_ids=segment_ids,
                use_one_hot_embeddings=use_one_hot_embeddings,
            )

        def r2to3(arr):
            return tf.reshape(arr, [batch_size, num_window, -1])

        # [Batch, num_window, window_length, hidden_size]
        pooled = model.get_pooled_output()
        logits = tf.keras.layers.Dense(1, name="cls_dense")(pooled) #
        self.logits = logits


        with tf.compat.v1.variable_scope(dual_model_prefix2):
            model = BertModel(
                config=config,
                is_training=is_training,
                input_ids=input_ids,
                input_mask=input_mask,
                token_type_ids=segment_ids,
                use_one_hot_embeddings=use_one_hot_embeddings,
            )
        logits = tf.keras.layers.Dense(1, name="cls_dense")(model.get_pooled_output())
        self.loss = 0

    def get_loss(self, label_ids_not_used):
        return self.loss

    def get_logits(self):
        return self.logits


class MES_hinge_layer1_load_many_loss(BertModelInterface):
    def __init__(self,
                 config,
                 is_training,
                 use_one_hot_embeddings=True,
                 features=None,
                 scope=None):

        super(MES_hinge_layer1_load_many_loss, self).__init__()
        alpha = config.alpha
        input_ids1 = features["input_ids1"]
        input_mask1 = features["input_mask1"]
        segment_ids1 = features["segment_ids1"]
        input_ids2 = features["input_ids2"]
        input_mask2 = features["input_mask2"]
        segment_ids2 = features["segment_ids2"]

        unit_length = config.max_seq_length
        d_seq_length = config.max_d_seq_length
        num_window = int(d_seq_length / unit_length)
        batch_size, _ = get_shape_list2(input_ids1)

        # [Batch, num_window, unit_seq_length]
        stacked_input_ids1, stacked_input_mask1, stacked_segment_ids1 = split_input(input_ids1,
                                                                                 input_mask1,
                                                                                 segment_ids1,
                                                                                 d_seq_length,
                                                                                 unit_length)
        # Ignore the window if
        # 1. The window is not first window and
        #   1.1 All input_mask is 0
        #   1.2 Content is too short, number of document tokens (other than query tokens) < 10

        # [Batch, num_window]
        is_first_window = tf.concat([tf.ones([batch_size, 1], tf.bool),
                                     tf.zeros([batch_size, num_window-1], tf.bool)], axis=1)
        num_content_tokens = tf.reduce_sum(stacked_segment_ids1, 2)
        has_enough_evidence = tf.less(10, num_content_tokens)
        is_valid_window = tf.logical_or(is_first_window, has_enough_evidence)
        pos_valid_window_mask = tf.cast(is_valid_window, tf.float32)

        self.is_first_window = is_first_window
        self.num_content_tokens = num_content_tokens
        self.has_enough_evidence = has_enough_evidence
        self.is_valid_window = is_valid_window

        with tf.compat.v1.variable_scope(dual_model_prefix1):
            model = BertModel(
                config=config,
                is_training=is_training,
                input_ids=r3to2(stacked_input_ids1),
                input_mask=r3to2(stacked_input_mask1),
                token_type_ids=r3to2(stacked_segment_ids1),
                use_one_hot_embeddings=use_one_hot_embeddings,
            )
            pooled = model.get_pooled_output()
            logits_2d = tf.keras.layers.Dense(1, name="cls_dense")(pooled)  #

        def r2to3(arr):
            return tf.reshape(arr, [batch_size, num_window, -1])

        # [Batch, num_window, window_length, hidden_size]
        logits_3d = r2to3(logits_2d) # [ batch, num_window, 1]
        pair_logits_layer1 = tf.reshape(logits_3d, [-1, num_window])


        # Probabilistic selection
        def select_seg(stacked_input_ids, indices):
            # indices : [batch_size, 1]
            return tf.gather(stacked_input_ids, indices, axis=1, batch_dims=1)

        valid_probs = pair_logits_layer1 * pos_valid_window_mask
        max_seg = tf.argmax(valid_probs, axis=1)
        pos_input_ids = select_seg(stacked_input_ids1, max_seg) # [batch_size, -1]
        pos_input_mask = select_seg(stacked_input_mask1, max_seg)
        pos_segment_ids = select_seg(stacked_segment_ids1, max_seg)

        def concat_pos_with_neg(pos_ids, neg_ids):
            #neg_ids : [batch_size, num_window, seq_length]
            pos_reshaped = tf.expand_dims(pos_ids, 1) # [, batch, seq_length]
            ids_3d = tf.concat([pos_reshaped, neg_ids], axis=1)
            return tf.reshape(ids_3d, [batch_size * (1+num_window), -1])

        stacked_input_ids2, stacked_input_mask2, stacked_segment_ids2 = split_input(input_ids2,
                                                                                 input_mask2,
                                                                                 segment_ids2,
                                                                                 d_seq_length,
                                                                                 unit_length)

        input_ids = concat_pos_with_neg(pos_input_ids, stacked_input_ids2)
        input_mask = concat_pos_with_neg(pos_input_mask, stacked_input_mask2)
        segment_ids = concat_pos_with_neg(pos_segment_ids, stacked_segment_ids2)
        print(input_ids)

        with tf.compat.v1.variable_scope(dual_model_prefix2):
            model = BertModel(
                config=config,
                is_training=is_training,
                input_ids=input_ids,
                input_mask=input_mask,
                token_type_ids=segment_ids,
                use_one_hot_embeddings=use_one_hot_embeddings,
            )
            logits = tf.keras.layers.Dense(1, name="cls_dense")(model.get_pooled_output())

        logits_3d = tf.reshape(logits, [batch_size, (1 + num_window), -1])
        pos_logits = logits_3d[:, 0, :]
        neg_logits = logits_3d[:, 1:, :]
        print(pos_logits)
        print(neg_logits)
        max_neg_logits = tf.reduce_max(neg_logits, axis=1)
        y_diff2 = pos_logits - max_neg_logits
        loss_arr = tf.maximum(1.0 - y_diff2, 0)
        self.logits = logits
        layer2_loss = tf.reduce_mean(loss_arr)
        self.loss = layer2_loss

    def get_loss(self):
        return self.loss

    def get_logits(self):
        return self.logits