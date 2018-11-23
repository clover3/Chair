
from models.transformer.transformer import *
from models.losses import *

label_smoothing = 0.1

class TransformerPairLM:
    def __init__(self, hp, voca_size, num_classes, is_training):
        # define decoder inputs
        self.x = tf.placeholder(dtype=tf.int32, shape=[None, hp.seq_max])
        self.y_cls = tf.placeholder(tf.int32, shape=(None,))
        self.y_seq = tf.placeholder(tf.int32, shape=(None, hp.seq_max))

        self.enc = transformer_encode(self.x, hp, voca_size, is_training)
        # Decoder
        sep_idx = hp.sent_max +1
        # Final linear projection
        self.seq_logits = tf.layers.dense(self.enc, voca_size)
        self.cls_logits = tf.layers.dense(self.enc[:,sep_idx,:], num_classes)
        self.preds = tf.to_int32(tf.argmax(self.cls_logits, axis=-1))

        if is_training:
            # Loss
            #            self.y_smoothed = label_smoothing(tf.one_hot(self.y, depth=voca_size))
            #            loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_smoothed)
            #            self.loss = tf.reduce_sum(loss * self.istarget) / (tf.reduce_sum(self.istarget))

            losses_seq, weight = padded_cross_entropy(self.seq_logits, self.y_seq, label_smoothing, reduce_sum=False)
            losses_cls = tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=self.cls_logits,
                labels=tf.one_hot(self.y_cls, num_classes))

            self.loss_seq_sum = tf.reduce_sum(tf.reduce_mean(losses_seq, axis=0))
            self.loss_seq_avg = tf.reduce_mean(losses_seq)

            self.loss_cls = tf.reduce_mean(losses_cls)
            self.loss = self.loss_cls + self.loss_seq_avg
            tf.summary.scalar('loss_seq', self.loss_seq_sum)
            tf.summary.scalar('loss_cls', self.loss_cls)
            tf.summary.scalar('loss', self.loss)
            tf.summary.merge_all()
