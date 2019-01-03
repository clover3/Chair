from task.transformer_est import Transformer, Classification
from models.transformer import bert
import tensorflow as tf
from data_generator.NLI import nli
from trainer import tf_module

class transformer_nli:
    def __init__(self, hp, voca_size):
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

        use_one_hot_embeddings = use_tpu
        is_training = True
        self.model = bert.BertModel(
            config=config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings)

        pred, loss = task.predict(self.model.get_sequence_output(), label_ids, is_training)

        self.logits = task.logits
        self.sout = tf.nn.softmax(self.logits)
        self.pred = pred
        self.loss = loss
        self.acc = task.acc
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('acc', self.acc)

        cl = tf.layers.dense(self.model.get_sequence_output(), 1, name="aux_conflict")
        cl = tf.reshape(cl, [-1, seq_length])
        cl = tf.nn.sigmoid(cl)
        #cl = tf.contrib.layers.layer_norm(cl)
        self.conf_logits = cl
        #self.pkc = self.conf_logits * self.rf_mask
        #rl_loss_list = tf.reduce_sum(self.pkc, axis=1)
        rl_loss_list = tf.reduce_sum(self.conf_logits * tf.cast(self.rf_mask, tf.float32), axis=1)


        num_tagged = tf.nn.relu(self.conf_logits+1)
        self.verbose_loss = tf.reduce_mean(tf.reduce_sum(num_tagged, axis=1))
        self.rl_loss = tf.reduce_mean(rl_loss_list)