import tensorflow as tf
from tlm.training.model_fn import get_bert_assignment_map
from models.transformer import bert_common_v2 as bc
from models.transformer import optimization_v2 as optimization


def get_seq_prediction(bert_config, input_tensor, positions, labels, label_weights):
    input_tensor = bc.gather_indexes(input_tensor, positions)

    hidden = bc.dense(bert_config.hidden,
                      bc.create_initializer(bert_config.initializer_range),
                      bc.get_activation(bert_config.hidden_act))(input_tensor)
    logits = bc.dense(1, bc.create_initializer(bert_config.initializer_range))(hidden)
    per_example_loss = tf.keras.losses.MSE(logits=logits, labels=labels)

    per_example_loss = tf.cast(label_weights, tf.float32) * per_example_loss
    losses = tf.reduce_sum(per_example_loss, axis=1)
    loss = tf.reduce_mean(losses)
    return loss, per_example_loss, logits


def sequence_prediction_model(bert_config, train_config, logging, model_class):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):    # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""
        logging.info("*** Features ***")
        for name in sorted(features.keys()):
            logging.info("name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_positions = features["label_positions"]
        labels = features["labels"]
        label_mask = features["label_mask"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        model = model_class(
                config=bert_config,
                is_training=is_training,
                input_ids=input_ids,
                input_mask=input_mask,
                token_type_ids=segment_ids,
                use_one_hot_embeddings=train_config.use_one_hot_embeddings,
        )

        (loss, example_loss, scores) = get_seq_prediction(bert_config, model.get_sequence_output(),
                                                       label_positions, labels, label_mask)

        total_loss = loss

        tvars = tf.compat.v1.trainable_variables()

        initialized_variable_names = {}
        scaffold_fn = None
        if train_config.init_checkpoint:
            if train_config.checkpoint_type == "":
                assignment_map, initialized_variable_names = get_bert_assignment_map(tvars, train_config.init_checkpoint)
                if train_config.use_tpu:
                    def tpu_scaffold():
                        tf.compat.v1.train.init_from_checkpoint(train_config.init_checkpoint, assignment_map)
                        return tf.compat.v1.train.Scaffold()

                    scaffold_fn = tpu_scaffold
                else:
                    tf.compat.v1.train.init_from_checkpoint(train_config.init_checkpoint, assignment_map)

        logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            logging.info("name = %s, shape = %s%s", var.name, var.shape, init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer_from_config(total_loss, train_config)
            output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
                    mode=mode,
                    loss=total_loss,
                    train_op=train_op,
                    scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:
            eval_metrics = (None, [])
            output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
                    mode=mode,
                    loss=total_loss,
                    eval_metrics=eval_metrics,
                    scaffold_fn=scaffold_fn)
        else:
            predictions = {
                    "input_ids": input_ids,
                    "loss": loss,
                    "example_loss": example_loss,
                    "scores": scores,
            }
            output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
                    mode=mode,
                    loss=total_loss,
                    predictions=predictions,
                    scaffold_fn=scaffold_fn)


        return output_spec

    return model_fn

