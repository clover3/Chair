import tensorflow as tf

from tlm.model.base import BertModelInterface, BertModel


def apply_weighted_loss(loss_arr, label_ids, alpha):
    print('label_ids', label_ids.shape)
    is_one = tf.cast(tf.equal(label_ids , 1), tf.float32)
    is_zero = tf.cast(tf.equal(label_ids, 0), tf.float32)
    print('is_one', is_one.shape)
    print("loss_arr", loss_arr.shape)

    postive_loss = alpha * is_one * loss_arr
    negative_loss = (1-alpha) * is_zero * loss_arr
    return postive_loss + negative_loss


class HingeBert(BertModelInterface):
    def __init__(self,
                 config,
                 is_training,
                 use_one_hot_embeddings=True,
                 features=None,
                 scope=None):
        super(HingeBert, self).__init__()
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]

        model = BertModel(
            config=config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings,
        )
        pooled = model.get_pooled_output()
        logits = tf.keras.layers.Dense(1, name="cls_dense")(pooled)

        self.prob = logits
        self.alpha = config.alpha

    def get_loss(self, label_ids):
        one_plus_minus_labels =  2. * tf.cast(label_ids, tf.float32) - 1
        losses = tf.keras.metrics.hinge(
            one_plus_minus_labels , self.prob
        )
        losses = apply_weighted_loss(losses, label_ids, self.alpha)
        loss = tf.reduce_mean(losses)
        self.loss = loss
        return loss

    def get_logits(self):
        return self.prob

