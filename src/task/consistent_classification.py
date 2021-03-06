from models.transformer.transformer import *
from trainer import tf_module

class ConsistentClassifier:
    def __init__(self, hp, voca_size, num_classes, is_training, feature_loc = 0):
        # define decoder inputs

        # Class 0 ~ num_classes-1  => label
        # y = num_classes implies that these items are from same users.
        self.x = tf.placeholder(dtype=tf.int32, shape=[None, hp.seq_max]) # Batch_size * Text_length
        self.y = tf.placeholder(tf.int32, shape=(None, ))

        self.x_pair = tf.placeholder(dtype=tf.int32, shape=[None, 2, hp.seq_max]) # Batch_size * Text_length
        x_pair_flat = tf.reshape(self.x_pair, [-1, hp.seq_max])

        with tf.variable_scope("consist_classifier"):
            enc = transformer_encode(self.x, hp, voca_size, is_training)
            self.logits = tf.layers.dense(enc[:,feature_loc,:], num_classes, name="cls_dense")

        with tf.variable_scope("consist_classifier", reuse=True):
            enc2 = transformer_encode(x_pair_flat, hp, voca_size, is_training)
            logits2 = tf.layers.dense(enc2[:,feature_loc,:], num_classes, name="cls_dense")
        #tf.summary.scalar('acc', self.acc)
        # Loss
        labels = tf.one_hot(self.y, num_classes)
        self.s_loss_arr = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.logits,
            labels=labels)
        self.acc = tf_module.accuracy(self.logits, self.y)


        logit_pair = tf.reshape(logits2, [-1, 2, num_classes])
        pred = tf.nn.softmax(logit_pair)
        self.pred_pair = pred

        def conflict_loss(pred):
            Pros = 1
            Against = 2
            l = pred[:,0,Pros] * tf.log(pred[:,1,Against]) + pred[:,0,Against] * tf.log(pred[:,1,Pros])
            return -l

        tau = 0.6
        #idk : I Don't Know
        def idk_loss(pred):
            s = 0
            for i in range(3):
                pred_no = pred[:,:,i]
                p = 0.33
                n_items = tf.reduce_sum(tf.ones_like(pred_no))
                mean = n_items * p
                dev = tf.sqrt(n_items * p * (1-p))
                normal = tf.distributions.Normal(loc=mean, scale=dev)
                l = -tf.log(normal.prob(tf.reduce_sum(pred_no)))
                s += tf.maximum(l-4, 0)
            return s


        self.idk_loss = idk_loss(pred)
        self.consist_loss = hp.alpha * (tf.reduce_mean(conflict_loss(pred)) + 0.2 * self.idk_loss)
        self.supervised_loss = (1-hp.alpha) * tf.reduce_mean(self.s_loss_arr)

        self.loss = self.supervised_loss + self.consist_loss

