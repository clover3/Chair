# This code is for tensorflow 2.0

import tensorflow as tf

import data_generator.NLI.nli_info
import tlm.model.base as bert
from trainer import tf_module


class Classification:
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def predict(self, enc, Y, is_train):
        if is_train:
            mode = tf.estimator.ModeKeys.TRAIN
        else:
            mode = tf.estimator.ModeKeys.EVAL
        return self.predict_ex(enc, Y, mode)

    def predict_ex(self, enc, Y, mode):
        feature_loc = 0
        logits = tf.compat.v1.layers.dense(enc[:,feature_loc,:], self.num_classes, name="cls_dense")
        labels = tf.one_hot(Y, self.num_classes)
        preds = tf.cast(tf.argmax(input=logits, axis=-1), dtype=tf.int32)
        self.acc = tf_module.accuracy(logits, Y)
        self.logits = logits
        if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
            self.loss_arr = tf.nn.softmax_cross_entropy_with_logits(
                logits=logits,
                labels=labels)
            self.loss = tf.reduce_mean(input_tensor=self.loss_arr)
            return preds, self.loss
        else:
            return preds


METHOD_CROSSENT = 2
METHOD_HINGE = 7

class transformer_nli:
    def __init__(self, hp,
                 input_ids,
                 input_mask,
                 segment_ids,
                 voca_size, method, is_training=True):
        config = bert.BertConfig(vocab_size=voca_size,
                                 hidden_size=hp.hidden_units,
                                 num_hidden_layers=hp.num_blocks,
                                 num_attention_heads=hp.num_heads,
                                 intermediate_size=hp.intermediate_size,
                                 type_vocab_size=hp.type_vocab_size,
                                 )

        seq_length = hp.seq_max
        use_tpu = False
        task = Classification(data_generator.NLI.nli_info.num_classes)

        label_ids = tf.compat.v1.placeholder(tf.int64, [None])
        if method in [0,1,3,4,5,6]:
            self.rf_mask = tf.compat.v1.placeholder(tf.float32, [None, seq_length])
        elif method in [METHOD_CROSSENT, METHOD_HINGE]:
            self.rf_mask = tf.compat.v1.placeholder(tf.int32, [None, seq_length])

        self.x_list = [input_ids, input_mask, segment_ids]
        self.y = label_ids

        use_one_hot_embeddings = use_tpu
        self.model = bert.BertModel(
            config=config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings)

        pred, loss = task.predict(self.model.get_sequence_output(), label_ids, True)

        self.logits = task.logits
        self.sout = tf.nn.softmax(self.logits)
        self.pred = pred
        self.loss = loss
        self.acc = task.acc
        if method == 0:
            cl = tf.compat.v1.layers.dense(self.model.get_sequence_output(), 1, name="aux_conflict")
            cl = tf.reshape(cl, [-1, seq_length])
            cl = tf.nn.sigmoid(cl)
            # cl = tf.contrib.layers.layer_norm(cl)
            self.conf_logits = cl
            # self.pkc = self.conf_logits * self.rf_mask
            # rl_loss_list = tf.reduce_sum(self.pkc, axis=1)
            rl_loss_list = tf.reduce_sum(input_tensor=self.conf_logits * tf.cast(self.rf_mask, tf.float32), axis=1)
            self.rl_loss = tf.reduce_mean(input_tensor=rl_loss_list)
        elif method == 1:
            cl = tf.compat.v1.layers.dense(self.model.get_sequence_output(), 1, name="aux_conflict")
            cl = tf.reshape(cl, [-1, seq_length])
            cl = tf.contrib.layers.layer_norm(cl)
            self.conf_logits = cl
            #rl_loss_list = tf_module.cossim(cl, self.rf_mask)
            #self.pkc = self.conf_logits * self.rf_mask
            rl_loss_list = tf.reduce_sum(input_tensor=self.conf_logits * self.rf_mask , axis=1)
            self.rl_loss = tf.reduce_mean(input_tensor=rl_loss_list)
        elif method == METHOD_CROSSENT:
            cl = tf.compat.v1.layers.dense(self.model.get_sequence_output(), 2, name="aux_conflict")
            probs = tf.nn.softmax(cl)
            losses = tf.compat.v1.losses.softmax_cross_entropy(onehot_labels=tf.one_hot(self.rf_mask, 2), logits=cl)
            self.conf_logits = probs[:,:,1] - 0.5
            self.rl_loss = tf.reduce_mean(input_tensor=losses)
        elif method == 3:
            cl = tf.compat.v1.layers.dense(self.model.get_sequence_output(), 1, name="aux_conflict")
            cl = tf.reshape(cl, [-1, seq_length])
            self.bias = tf.Variable(0.0)
            self.conf_logits = (cl + self.bias)
            rl_loss_list = tf.nn.relu(1 - self.conf_logits * self.rf_mask)
            rl_loss_list = tf.reduce_mean(input_tensor=rl_loss_list, axis=1)
            self.rl_loss = tf.reduce_mean(input_tensor=rl_loss_list)
            labels = tf.greater(self.rf_mask, 0)
            hinge_losses = tf.compat.v1.losses.hinge_loss(labels, self.conf_logits)
            self.hinge_loss = tf.reduce_sum(input_tensor=hinge_losses)
        elif method == 4:
            cl = tf.compat.v1.layers.dense(self.model.get_sequence_output(), 1, name="aux_conflict")
            cl = tf.reshape(cl, [-1, seq_length])
            cl = tf.contrib.layers.layer_norm(cl)
            self.conf_logits = cl
            labels = tf.greater(self.rf_mask, 0)
            hinge_losses = tf.compat.v1.losses.hinge_loss(labels, self.conf_logits)
            self.rl_loss = hinge_losses
        elif method == 5:
            cl = tf.compat.v1.layers.dense(self.model.get_sequence_output(), 1, name="aux_conflict")
            cl = tf.reshape(cl, [-1, seq_length])
            #cl = tf.contrib.layers.layer_norm(cl)
            self.conf_logits = cl
            self.labels = tf.cast(tf.greater(self.rf_mask, 0), tf.float32)
            self.rl_loss = tf.reduce_mean(input_tensor=tf_module.correlation_coefficient_loss(cl, -self.rf_mask))
        elif method == 6:
            cl = tf.compat.v1.layers.dense(self.model.get_sequence_output(), 1, name="aux_conflict")
            #cl = tf.layers.dense(cl1, 1, name="aux_conflict2")
            cl = tf.reshape(cl, [-1, seq_length])
            #cl = tf.nn.sigmoid(cl)
            #cl = tf.contrib.layers.layer_norm(cl)
            self.conf_logits = cl
            #rl_loss_list = tf.reduce_sum(self.conf_logits * self.rf_mask , axis=1)
            self.rl_loss = tf.reduce_mean(input_tensor=tf_module.correlation_coefficient_loss(cl, -self.rf_mask))
        elif method == METHOD_HINGE:
            cl = tf.compat.v1.layers.dense(self.model.get_sequence_output(), 1, name="aux_conflict")
            cl = tf.reshape(cl, [-1, seq_length])
            self.conf_logits = cl
            labels = tf.greater(self.rf_mask, 0)
            hinge_losses = tf.compat.v1.losses.hinge_loss(labels, self.conf_logits)
            self.rl_loss = tf.reduce_sum(input_tensor=hinge_losses)

        self.conf_softmax = tf.nn.softmax(self.conf_logits, axis=-1)
