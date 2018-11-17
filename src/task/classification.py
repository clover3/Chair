from models.transformer.transformer import *
from trainer import tf_module


class TransformerClassifier:
    def __init__(self, hp, voca_size, num_classes, is_training):
        # define decoder inputs
        self.x = tf.placeholder(dtype=tf.int32, shape=[None, hp.seq_max]) # Batch_size * Text_length
        self.y = tf.placeholder(tf.int32, shape=(None, ))

        self.enc = transformer_encode(self.x, hp, voca_size, is_training)
        # Decoder
        # Final linear projection
        self.logits = tf.layers.dense(self.enc[:,0,:], num_classes)
        self.acc = tf_module.accuracy(self.logits, self.y)
        #tf.summary.scalar('acc', self.acc)

        if is_training:
            # Loss
            labels = tf.one_hot(self.y, num_classes)
            self.loss_arr = tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=self.logits,
                labels=labels)
            self.loss = tf.reduce_mean(self.loss_arr)
            #tf.summary.scalar('loss', self.loss)

