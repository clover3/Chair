import tensorflow as tf

# Final probability is mean of each segments
from models.transformer.bert_common_v2 import get_shape_list2
from tlm.model.base import BertModelInterface, BertModel
from tlm.model.dual_model_common import dual_model_prefix1, dual_model_prefix2
from tlm.model.multiple_evidence import split_input, r3to2


# Segment wise
class MES_sel(BertModelInterface):
    def __init__(self,
                 config,
                 is_training,
                 use_one_hot_embeddings=True,
                 features=None,
                 scope=None):

        super(MES_sel, self).__init__()
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
        logits_2d = tf.keras.layers.Dense(2, name="cls_dense")(pooled) #
        logits_3d = r2to3(logits_2d)
        label_ids_repeat = tf.tile(label_ids, [1, num_window])
        loss_arr = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits_3d,
            labels=label_ids_repeat)
        layer1_loss = tf.reduce_mean(loss_arr)

        probs = tf.nn.softmax(logits_3d)[:, :, 1]  # [batch_size, num_window]

        # Probabilistic selection
        def select_seg(stacked_input_ids, indices):
            # indices : [batch_size, 1]
            return tf.gather(stacked_input_ids, indices, axis=1, batch_dims=1)

        max_seg = tf.argmax(probs, axis=1)
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
        alpha = 0.1
        loss = alpha * layer1_loss + layer2_loss
        self.loss = loss


    def get_loss(self, label_ids_not_used):
        return self.loss

    def get_logits(self):
        return self.logits


# Segment wise
class MES_sel_pred(BertModelInterface):
    def __init__(self,
                 config,
                 is_training,
                 use_one_hot_embeddings=True,
                 features=None,
                 scope=None):

        super(MES_sel_pred, self).__init__()
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


class MES_pad(BertModelInterface):
    def __init__(self,
                 config,
                 is_training,
                 use_one_hot_embeddings=True,
                 features=None,
                 scope=None):

        super(MES_pad, self).__init__()
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]

        trained_l = config.trained_seq_length
        data_l = config.data_seq_length

        batch_size, _ = get_shape_list2(input_ids)

        add_len = trained_l - data_l
        zero_pad = tf.zeros([batch_size, add_len], tf.int32)
        input_ids = tf.concat([input_ids, zero_pad], axis=1)
        input_mask = tf.concat([input_mask, zero_pad], axis=1)
        segment_ids = tf.concat([segment_ids, zero_pad], axis=1)

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


# Segment wise
class MES_sel_pred_rescope(BertModelInterface):
    def __init__(self,
                 config,
                 is_training,
                 use_one_hot_embeddings=True,
                 features=None,
                 scope=None):

        super(MES_sel_pred_rescope, self).__init__()
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