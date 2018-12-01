
from models.transformer.transformer import *
from models.losses import *
from models.transformer.modules import gelu_fast
from trainer import tf_module

class PairFeature:
    def __init__(self, hp, voca_size, is_training):
        # define decoder inputs
        input_len = hp.seq_max * 2 + 1
        self.x = tf.placeholder(dtype=tf.int32, shape=[None, input_len])
        self.y_cls = tf.placeholder(tf.int32, shape=(None,))

        sent1 = self.x[:, :hp.sent_max]
        sent2 = self.x[:,hp.sent_max+1:]

        def extract_feature(sent):
            enc = transformer_encode(sent, hp, voca_size, is_training)
            return tf.layers.dense(enc[:,0], hp.feature_size)

        with tf.variable_scope("feature_encoder"):
            feature1 = extract_feature(sent1)
        with tf.variable_scope("feature_encoder", reuse=True):
            feature2 = extract_feature(sent2)
        # Decoder
        # Final linear projection
        self.feature1 = feature1
        self.feature2 = feature2

        combine = tf.multiply(feature1, feature2)
        combine = gelu_fast(combine)
        pred = tf.reshape(tf.layers.dense(combine, 1), [-1])

        if is_training:
            # Loss
            #            self.y_smoothed = label_smoothing(tf.one_hot(self.y, depth=voca_size))
            #            loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_smoothed)
            #            self.loss = tf.reduce_sum(loss * self.istarget) / (tf.reduce_sum(self.istarget))

            losses_cls = tf.losses.absolute_difference(pred, self.y_cls)

            self.loss = tf.reduce_mean(losses_cls)
            tf.summary.scalar('loss', self.loss)
            tf.summary.merge_all()


class PairFeatureClassification:
    def __init__(self, hp, voca_size, num_classes, is_training):
        # define decoder inputs
        input_len = hp.seq_max
        self.x = tf.placeholder(dtype=tf.int32, shape=[None, input_len])
        self.y = tf.placeholder(tf.int32, shape=(None, ))


        def extract_feature(sent):
            enc = transformer_encode(sent, hp, voca_size, is_training)
            return tf.layers.dense(enc[:,0], hp.feature_size)

        with tf.variable_scope("feature_encoder"):
            feature1 = extract_feature(self.x)


        self.feature1 = feature1
        self.logits = tf.layers.dense(feature1, num_classes)
        #tf.summary.scalar('acc', self.acc)
        self.acc = tf_module.accuracy(self.logits, self.y)

        # Loss
        labels = tf.one_hot(self.y, num_classes)
        self.loss_arr = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.logits,
            labels=labels)
        self.loss = tf.reduce_mean(self.loss_arr)

