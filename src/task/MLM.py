
from models.transformer.transformer import *
from models.losses import *

label_smoothing = 0.1

class TransformerLM:
    def __init__(self, hp, voca_size, is_training):
        # define decoder inputs
        self.x = tf.placeholder(dtype=tf.int32, shape=[None, hp.seq_max])
        self.y = tf.placeholder(tf.int32, shape=(None, hp.seq_max))

        self.enc = transformer_encode(self.x, hp, voca_size, is_training)
        # Decoder

        # Final linear projection
        self.logits = tf.layers.dense(self.enc, voca_size)
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
