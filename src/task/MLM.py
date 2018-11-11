
from models.transformer.transformer import *

class TransformerLM:
    def __init__(self, hp, voca_size, is_training):
        # define decoder inputs
        self.x = tf.placeholder(dtype=tf.int32, shape=[None, hp.seq_max])
        self.y = tf.placeholder(tf.int32, shape=(None, hp.seq_max))

        self.enc = transformer_encode(self.x, hp, voca_size, is_training)
        # Decoder

        # Final linear projection
        self.logits = tf.layers.dense(self.enc, voca_size)
        self.preds = tf.to_int32(tf.arg_max(self.logits, dimension=-1))
        self.istarget = tf.to_float(tf.not_equal(self.y, 0))
        self.acc = tf.reduce_sum(tf.to_float(tf.equal(self.preds, self.y)) * self.istarget) / (tf.reduce_sum(self.istarget))
        tf.summary.scalar('acc', self.acc)

        if is_training:
            # Loss
            self.y_smoothed = label_smoothing(tf.one_hot(self.y, depth=voca_size))
            self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_smoothed)
            self.mean_loss = tf.reduce_sum(self.loss * self.istarget) / (tf.reduce_sum(self.istarget))
            tf.summary.scalar('mean_loss', self.mean_loss)
