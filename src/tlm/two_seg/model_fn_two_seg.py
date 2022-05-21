from models.transformer.bert_common_v2 import get_shape_list2
from my_tf import tf
from tlm.model.base import BertModelInterface, BertModel
from trainer_v2.custom_loop.modeling_common.network_utils import vector_three_feature


# Segment wise
class CLSConcat(BertModelInterface):
    def __init__(self,
                 config,
                 is_training,
                 use_one_hot_embeddings=True,
                 features=None,
                 scope=None):

        super(CLSConcat, self).__init__()
        input_ids1 = features["input_ids0"]
        input_mask1 = features["input_mask0"]
        segment_ids1 = features["segment_ids0"]
        input_ids2 = features["input_ids1"]
        input_mask2 = features["input_mask1"]
        segment_ids2 = features["segment_ids1"]

        l1 = config.max_seq_length
        l2 = config.max_seq_length2
        pad_len = l1 - l2
        def pad(ids):
            return tf.pad(ids, [(0, 0), (0, pad_len)])

        input_ids = tf.concat([input_ids1, pad(input_ids2)], axis=0)
        input_mask = tf.concat([input_mask1, pad(input_mask2)], axis=0)
        segment_ids = tf.concat([segment_ids1, pad(segment_ids2)], axis=0)
        label_ids = features["label_ids"]
        self.labels = label_ids

        batch_size, _ = get_shape_list2(input_ids1)

        # [Batch, num_window, unit_seq_length]
        model = BertModel(
            config=config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings,
        )
        pooled = model.get_pooled_output()
        print(pooled)
        pair_pooled\
            = tf.reshape(pooled, [2, batch_size, -1])
        cls1 = pair_pooled[0, :]
        cls2 = pair_pooled[1, :]
        feature_rep = vector_three_feature(cls1, cls2)
        hidden = tf.keras.layers.Dense(config.hidden_size, activation='relu')(feature_rep)
        logits = tf.keras.layers.Dense(config.num_classes)(hidden)
        self.logits = logits
        label_ids = tf.reshape(label_ids, [-1])
        loss_arr = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits,
            labels=label_ids)

        self.loss = tf.reduce_mean(loss_arr)

    def get_logits(self):
        return self.logits

    def get_loss(self):
        return self.loss

    def get_metric(self):
        def metric_fn(logits, label):
            """Computes the loss and accuracy of the model."""
            pred = tf.argmax(
                input=logits, axis=-1, output_type=tf.int32)

            label = tf.reshape(label, [-1])
            accuracy = tf.compat.v1.metrics.accuracy(labels=label, predictions=pred)
            return {'accuracy': accuracy}

        return (metric_fn, [self.logits, self.labels])