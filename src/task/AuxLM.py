from models.transformer.transformer import *
from models.losses import *
from trainer import tf_module
label_smoothing = 0.1

class AuxLM:
    def __init__(self, hp, hp_aux, voca_size, is_training):
        # define decoder inputs
        self.x = tf.placeholder(dtype=tf.int32, shape=[None, hp.seq_max])
        self.y = tf.placeholder(tf.int32, shape=(None, hp.seq_max))

        with tf.variable_scope("aux"):
            self.y_aux = tf.placeholder(tf.int32, shape=(None,))
            num_classes = 3
            enc_aux = transformer_encode(self.x, hp_aux, voca_size, is_training)
            aux_logit = tf.layers.dense(enc_aux[:, 0, :], num_classes, name="cls_dense")
            labels = tf.one_hot(self.y_aux, num_classes)
            loss_arr = tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=aux_logit,
                labels=labels)
            self.aux_loss = tf.reduce_mean(loss_arr)
            self.aux_acc = tf_module.accuracy(aux_logit, self.y_aux)

            aux_v = tf.reshape(aux_logit, [-1, 1, num_classes])
            added_dim = hp.hidden_units - num_classes
            aux_v = tf.pad(aux_v, [(0,0),(0,0),(0,added_dim)])
        self.enc = transformer_aux(self.x, hp, voca_size, is_training, aux_v)
        # Decoder

        # Final linear projection
        self.logits = tf.layers.dense(self.enc[:,:-1], voca_size)
        self.preds = tf.to_int32(tf.argmax(self.logits, axis=-1))
        self.istarget = tf.to_float(tf.not_equal(self.y, 0))
        self.acc = tf.reduce_sum(tf.to_float(tf.equal(self.preds, self.y)) * self.istarget) / (tf.reduce_sum(self.istarget))
        tf.summary.scalar('acc', self.acc)

        if is_training:
            # Loss

            loss_list, weight = padded_cross_entropy(self.logits, self.y, label_smoothing,
                                                     reduce_sum=False)
            self.loss = tf.reduce_mean(loss_list)
            tf.summary.scalar('loss', self.loss)
