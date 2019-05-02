from models.transformer import bert
import tensorflow as tf
import collections
import re

def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
    """Compute the union of the current variables and checkpoint variables."""
    assignment_map = {}
    initialized_variable_names = {}

    name_to_variable = collections.OrderedDict()
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var

    init_vars = tf.train.list_variables(init_checkpoint)

    assignment_map = collections.OrderedDict()
    for x in init_vars:
        (name, var) = (x[0], x[1])
        if name not in name_to_variable:
            continue
        assignment_map[name] = name
        initialized_variable_names[name] = 1
        initialized_variable_names[name + ":0"] = 1

    return (assignment_map, initialized_variable_names)


class TransformerEst:
    def __init__(self, hp, voca_size, task, use_tpu=False, init_checkpoint=None):
        self.hp = hp
        self.voca_size = voca_size
        self.task = task
        self.dtype = tf.float32
        self.use_tpu = use_tpu
        self.use_one_hot_embeddings = use_tpu
        self.init_checkpoint = init_checkpoint


    def model_fn(self, mode, features, labels, params):
        # train mode: required loss and train_op
        # eval mode: required loss
        # predict mode: required predictions
        eval_metrics = None
        train_op = None
        if mode == tf.estimator.ModeKeys.TRAIN :
            predictions, loss = self.network(features, mode)
            train_op = self.get_train_op(loss)
        elif mode == tf.estimator.ModeKeys.EVAL:
            predictions, loss = self.network(features, mode)

            def metric_fn(logits, loss_arr, label_ids):
                predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                accuracy = tf.metrics.accuracy(label_ids, predictions)
                loss = tf.metrics.mean(loss_arr)

                return {
                    "eval_acc": accuracy,
                    "eval_loss": loss,
                }
            eval_metrics = (metric_fn, [self.task.logits, self.task.loss_arr, self.label_ids])
        else:
            predictions = self.network(features, mode)
            loss = 1

        tvars = tf.trainable_variables()
        initialized_variable_names = None
        scaffold_fn = None
        if self.init_checkpoint:
            (assignment_map, initialized_variable_names) = get_assignment_map_from_checkpoint(tvars, self.init_checkpoint)
            if self.use_tpu:
                def tpu_scaffold():
                    tf.train.init_from_checkpoint(self.init_checkpoint, assignment_map)
                    return tf.train.Scaffold()
                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(self.init_checkpoint, assignment_map)
        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        return tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,
            eval_metrics=eval_metrics,
            scaffold_fn=scaffold_fn,
            predictions={"prediction": predictions})

    def network(self, features, mode):
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
        self.label_ids = features["label_ids"]

        is_training = (tf.estimator.ModeKeys.TRAIN == mode)
        self.model = bert.BertModel(
            config=config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=self.use_one_hot_embeddings)

        enc = self.model.get_sequence_output()
        return self.task.predict_ex(enc, self.label_ids, mode)

    def get_train_op(self, loss):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.hp.lr, beta1=0.9, beta2=0.98, epsilon=1e-8)
        if self.use_tpu:
            optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
        return train_op

