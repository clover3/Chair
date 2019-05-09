from task.transformer_est import Transformer, Classification
from models.transformer import bert
import tensorflow as tf
from data_generator.NLI import nli
from trainer import tf_module

class transformer_nli:
    def __init__(self, hp, voca_size, method, is_training=True):
        config = bert.BertConfig(vocab_size=voca_size,
                                 hidden_size=hp.hidden_units,
                                 num_hidden_layers=hp.num_blocks,
                                 num_attention_heads=hp.num_heads,
                                 intermediate_size=hp.intermediate_size,
                                 type_vocab_size=hp.type_vocab_size,
                                 )

        seq_length = hp.seq_max
        use_tpu = False
        task = Classification(nli.num_classes)

        input_ids = tf.placeholder(tf.int64, [None, seq_length])
        input_mask = tf.placeholder(tf.int64, [None, seq_length])
        segment_ids = tf.placeholder(tf.int64, [None, seq_length])
        label_ids = tf.placeholder(tf.int64, [None])
        if method in [0,1,3,4,5,6]:
            self.rf_mask = tf.placeholder(tf.float32, [None, seq_length])
        elif method in [2]:
            self.rf_mask = tf.placeholder(tf.int32, [None, seq_length])

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
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('acc', self.acc)
        if method == 0:
            cl = tf.layers.dense(self.model.get_sequence_output(), 1, name="aux_conflict")
            cl = tf.reshape(cl, [-1, seq_length])
            cl = tf.nn.sigmoid(cl)
            # cl = tf.contrib.layers.layer_norm(cl)
            self.conf_logits = cl
            # self.pkc = self.conf_logits * self.rf_mask
            # rl_loss_list = tf.reduce_sum(self.pkc, axis=1)
            rl_loss_list = tf.reduce_sum(self.conf_logits * tf.cast(self.rf_mask, tf.float32), axis=1)
            self.rl_loss = tf.reduce_mean(rl_loss_list)
        elif method == 1:
            cl = tf.layers.dense(self.model.get_sequence_output(), 1, name="aux_conflict")
            cl = tf.reshape(cl, [-1, seq_length])
            cl = tf.contrib.layers.layer_norm(cl)
            self.conf_logits = cl
            #rl_loss_list = tf_module.cossim(cl, self.rf_mask)
            #self.pkc = self.conf_logits * self.rf_mask
            rl_loss_list = tf.reduce_sum(self.conf_logits * self.rf_mask , axis=1)
            self.rl_loss = tf.reduce_mean(rl_loss_list)
        elif method == 2:
            cl = tf.layers.dense(self.model.get_sequence_output(), 2, name="aux_conflict")
            probs = tf.nn.softmax(cl)
            losses = tf.losses.softmax_cross_entropy(onehot_labels=tf.one_hot(self.rf_mask, 2), logits=cl)
            self.conf_logits = probs[:,:,1] - 0.5
            self.rl_loss = tf.reduce_mean(losses)
        elif method == 3:
            cl = tf.layers.dense(self.model.get_sequence_output(), 1, name="aux_conflict")
            cl = tf.reshape(cl, [-1, seq_length])
            self.bias = tf.Variable(0.0)
            self.conf_logits = (cl + self.bias)
            rl_loss_list = tf.nn.relu(1 - self.conf_logits * self.rf_mask)
            rl_loss_list = tf.reduce_mean(rl_loss_list, axis=1)
            self.rl_loss = tf.reduce_mean(rl_loss_list)
            labels = tf.greater(self.rf_mask, 0)
            hinge_losses = tf.losses.hinge_loss(labels, self.conf_logits)
            self.hinge_loss = tf.reduce_sum(hinge_losses)
        elif method == 4:
            cl = tf.layers.dense(self.model.get_sequence_output(), 1, name="aux_conflict")
            cl = tf.reshape(cl, [-1, seq_length])
            cl = tf.contrib.layers.layer_norm(cl)
            self.conf_logits = cl
            labels = tf.greater(self.rf_mask, 0)
            hinge_losses = tf.losses.hinge_loss(labels, self.conf_logits)
            self.rl_loss = hinge_losses
        elif method == 5:
            cl = tf.layers.dense(self.model.get_sequence_output(), 1, name="aux_conflict")
            cl = tf.reshape(cl, [-1, seq_length])
            #cl = tf.contrib.layers.layer_norm(cl)
            self.conf_logits = cl
            self.labels = tf.cast(tf.greater(self.rf_mask, 0), tf.float32)
            self.rl_loss = tf.reduce_mean(tf_module.correlation_coefficient_loss(cl, -self.rf_mask))
        elif method == 6:
            cl = tf.layers.dense(self.model.get_sequence_output(), 1, name="aux_conflict")
            #cl = tf.layers.dense(cl1, 1, name="aux_conflict2")
            cl = tf.reshape(cl, [-1, seq_length])
            #cl = tf.nn.sigmoid(cl)
            #cl = tf.contrib.layers.layer_norm(cl)
            self.conf_logits = cl
            #rl_loss_list = tf.reduce_sum(self.conf_logits * self.rf_mask , axis=1)
            self.rl_loss = tf.reduce_mean(tf_module.correlation_coefficient_loss(cl, -self.rf_mask))

#            self.rl_loss = tf.reduce_mean(rl_loss_list)
            #with tf.device("/device:GPU:1"):
            #    pl = tf.layers.dense(self.model.get_sequence_output(), hp.hidden_units, name="aux_pairing1")
            #    pl = tf.layers.dense(pl, 1, name="aux_pairing2")
            #    pl = tf.reshape(pl, [-1, seq_length])
            #    pl = tf.contrib.layers.layer_norm(pl)
            #    self.pair_logits = pl
            #    self.pr_mask = tf.placeholder(tf.float32, [None, seq_length])
            #    labels = tf.greater(self.pr_mask, 0)
            #    #hinge_losses = tf.losses.hinge_loss(labels, self.pair_logits)
            #    pr_loss_list = tf.reduce_sum(self.pair_logits * self.pr_mask, axis=1)
            #    #pr_loss_list = hinge_losses
            #    self.pr_loss = tf.reduce_mean(pr_loss_list)



class transformer_nli_embedding_in:
    def __init__(self, hp, voca_size, is_training):
        config = bert.BertConfig(vocab_size=voca_size,
                                 hidden_size=hp.hidden_units,
                                 num_hidden_layers=hp.num_blocks,
                                 num_attention_heads=hp.num_heads,
                                 intermediate_size=hp.intermediate_size,
                                 type_vocab_size=hp.type_vocab_size,
                                 )

        seq_length = hp.seq_max
        use_tpu = False
        task = Classification(nli.num_classes)

        input_ids = tf.placeholder(tf.int64, [None, seq_length])
        input_mask = tf.placeholder(tf.int64, [None, seq_length])
        segment_ids = tf.placeholder(tf.int64, [None, seq_length])
        label_ids = tf.placeholder(tf.int64, [None])
#        self.rf_mask = tf.placeholder(tf.float32, [None, seq_length])
        self.rf_mask = tf.placeholder(tf.int32, [None, seq_length])

        self.x_list = [input_ids, input_mask, segment_ids]
        self.y = label_ids
        self.encoded_embedding_in = tf.placeholder(tf.float32, [None, seq_length, hp.hidden_units])
        self.attention_mask_in = tf.placeholder(tf.float32, [None, seq_length, seq_length])
        use_one_hot_embeddings = use_tpu
        self.model = bert.BertEmbeddingInOut(
            config=config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings,
            embeddding_as_input=(self.encoded_embedding_in, self.attention_mask_in),
        )

        self.encoded_embedding_out = self.model.embedding_output
        self.attention_mask_out = self.model.attention_mask

        pred, loss = task.predict(self.model.get_sequence_output(), label_ids, True)

        self.logits = task.logits
        self.sout = tf.nn.softmax(self.logits)
        self.pred = pred
        self.loss = loss
        self.acc = task.acc
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('acc', self.acc)

        cl = tf.layers.dense(self.model.get_sequence_output(), 1, name="aux_conflict")
        cl = tf.reshape(cl, [-1, seq_length])
        #cl = tf.nn.sigmoid(cl)
        #cl = tf.contrib.layers.layer_norm(cl)
        self.conf_logits = cl
        #self.pkc = self.conf_logits * self.rf_mask
        #rl_loss_list = tf.reduce_sum(self.pkc, axis=1)
        rl_loss_list = tf.reduce_sum(self.conf_logits * tf.cast(self.rf_mask, tf.float32), axis=1)

        num_tagged = tf.nn.relu(self.conf_logits+1)
        self.verbose_loss = tf.reduce_mean(tf.reduce_sum(num_tagged, axis=1))
        self.rl_loss = tf.reduce_mean(rl_loss_list)

