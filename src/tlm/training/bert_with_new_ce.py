import tensorflow as tf

from tlm.model.base import BertModelInterface, BertModel


def new_ce(y, logits, alpha):
    losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.one_hot(y, 2))
    c = -tf.math.log(alpha)
    losses = tf.maximum(losses, c)
    return losses


class BertNewCE(BertModelInterface):
    def __init__(self,
                 config,
                 is_training,
                 use_one_hot_embeddings=True,
                 features=None,
                 scope=None):
        super(BertNewCE, self).__init__()
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
        logits = tf.keras.layers.Dense(2, name="cls_dense")(pooled)

        self.logits = logits
        self.alpha = config.alpha

    def get_loss(self, label_ids):
        losses = new_ce(label_ids, self.logits, self.alpha)
        loss = tf.reduce_mean(losses)
        self.loss = loss
        return loss

    def get_logits(self):
        return self.logits

