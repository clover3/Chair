from models.transformer import bert
import tensorflow as tf



class TransformerEst:
    def __init__(self, hp, voca_size, task, use_tpu=False):
        self.hp = hp
        self.voca_size = voca_size
        self.task = task
        self.dtype = tf.float32
        self.use_tpu = use_tpu
        self.use_one_hot_embeddings = use_tpu



    def model_fn(self, mode, features, labels, params):

        # train mode: required loss and train_op
        # eval mode: required loss
        # predict mode: required predictions
        metrics = {}
        train_op = None
        if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
            predictions, loss = self.predict(features, True)
            train_op = self.get_train_op(loss)
            metrics = {}
            #tf.summary.scalar("accuracy", self.task.acc)
        else:
            predictions = self.predict(features, False)
            loss = 1

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=metrics,
            predictions={"prediction": predictions})

    def predict(self, features, is_training):
        config = bert.BertConfig(vocab_size=self.voca_size,
                                 hidden_size=self.hp.hidden_units,
                                 num_hidden_layers=self.hp.num_blocks,
                                 num_attention_heads=self.hp.num_heads,
                                 intermediate_size=self.hp.intermediate_size,
                                 type_vocab_size=self.hp.type_vocab_size,
                                 )
        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]

        self.model = bert.BertModel(
            config=config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=self.use_one_hot_embeddings)

        enc = self.model.get_sequence_output()
        return self.task.predict(enc, label_ids, is_training)

    def get_train_op(self, loss):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.hp.lr, beta1=0.9, beta2=0.98, epsilon=1e-8)
        if self.use_tpu:
            optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return train_op

