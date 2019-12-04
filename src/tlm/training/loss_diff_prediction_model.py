import tensorflow as tf

from models.transformer import bert_common_v2 as bc
from models.transformer import optimization_v2 as optimization
from tf_util.tf_logging import tf_logging
from tlm.training.input_fn_common import get_lm_basic_features, get_lm_mask_features, format_dataset
from tlm.training.model_fn import get_bert_assignment_map


def loss_to_prob_pair(loss):
    y0 = tf.exp(-loss)
    y1 = 1 - y0
    return tf.stack([y0, y1], -1)


def get_loss_independently(bert_config, input_tensor,
                           masked_lm_positions, masked_lm_weights, loss_base, loss_target):
    input_tensor = bc.gather_indexes(input_tensor, masked_lm_positions)

    hidden = bc.dense(bert_config.hidden_size,
                      bc.create_initializer(bert_config.initializer_range),
                      bc.get_activation(bert_config.hidden_act))(input_tensor)


    def get_regression_and_loss(hidden_vector, loss_label):
        logits = bc.dense(2, bc.create_initializer(bert_config.initializer_range))(hidden_vector)
        print("logits", logits.shape)
        gold_prob = loss_to_prob_pair(loss_label)
        print("gold_prob", gold_prob.shape)
        logits = tf.reshape(logits, gold_prob.shape)

        per_example_loss = tf.nn.softmax_cross_entropy_with_logits(
            gold_prob,
            logits,
            axis=-1,
            name=None
        )
        per_example_loss = tf.cast(masked_lm_weights, tf.float32) * per_example_loss
        losses = tf.reduce_sum(per_example_loss, axis=1)
        loss = tf.reduce_mean(losses)

        return loss, per_example_loss

    loss1, per_example_loss1 = get_regression_and_loss(hidden, loss_base)
    loss2, per_example_loss2 = get_regression_and_loss(hidden, loss_target)

    total_loss = loss1 + loss2
    return total_loss, loss1, loss2, per_example_loss1, per_example_loss2


def get_diff_loss(bert_config, input_tensor,
                           masked_lm_positions, masked_lm_weights, loss_base, loss_target):
    input_tensor = bc.gather_indexes(input_tensor, masked_lm_positions)

    hidden = bc.dense(bert_config.hidden_size,
                      bc.create_initializer(bert_config.initializer_range),
                      bc.get_activation(bert_config.hidden_act))(input_tensor)

    logits = bc.dense(1, bc.create_initializer(bert_config.initializer_range))(hidden)

    base_prob = tf.exp(-loss_base)
    target_prob = tf.exp(-loss_target)

    prob_diff = base_prob - target_prob

    per_example_loss = tf.losses.MAE(prob_diff, logits)
    per_example_loss = tf.cast(masked_lm_weights, tf.float32) * per_example_loss
    losses = tf.reduce_sum(per_example_loss, axis=1)
    loss = tf.reduce_mean(losses)

    return loss, per_example_loss, logits



def recover_mask(input_ids, masked_lm_positions, masked_lm_ids):
    batch_size = input_ids.shape[0]
    seq_length = input_ids.shape[1]
    flat_input_ids = tf.reshape(input_ids, [-1])
    offsets = tf.range(batch_size) * seq_length
    masked_lm_positions = masked_lm_positions + tf.expand_dims(offsets, 1)
    masked_lm_positions = tf.reshape(masked_lm_positions, [-1, 1])
    masked_lm_ids = tf.reshape(masked_lm_ids, [-1])

    output = tf.tensor_scatter_nd_update(flat_input_ids, masked_lm_positions, masked_lm_ids)
    output = tf.reshape(output, [batch_size, seq_length])
    output = tf.concat([input_ids[:, :1], output[:,1:]], axis=1)
    return output


def loss_diff_prediction_model(bert_config, train_config, model_class, model_config):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):    # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""
        tf_logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf_logging.info("name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        loss_base = features["loss_base"]
        loss_target = features["loss_target"]
        masked_lm_positions = features["masked_lm_positions"]
        masked_lm_ids = features["masked_lm_ids"]
        masked_lm_weights = features["masked_lm_weights"]

        input_ids = recover_mask(input_ids, masked_lm_positions, masked_lm_ids)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        model = model_class(
                config=bert_config,
                is_training=is_training,
                input_ids=input_ids,
                input_mask=input_mask,
                token_type_ids=segment_ids,
                use_one_hot_embeddings=train_config.use_one_hot_embeddings,
        )

        if model_config.loss_model == "independent":
            total_loss, loss1, loss2, per_example_loss1, per_example_loss2 \
                = get_loss_independently(bert_config, model.get_sequence_output(),
                                         masked_lm_positions, masked_lm_weights, loss_base, loss_target)
            tf.summary.scalar("loss_base", loss1)
            tf.summary.scalar("loss_target", loss2)
        elif model_config.loss_model == "diff_regression":
            total_loss, losses, logits = get_diff_loss(bert_config, model.get_sequence_output(),
                                        masked_lm_positions, masked_lm_weights, loss_base, loss_target)

        tvars = tf.compat.v1.trainable_variables()

        initialized_variable_names = {}
        scaffold_fn = None
        if train_config.init_checkpoint:
            assignment_map, initialized_variable_names = get_bert_assignment_map(tvars, train_config.init_checkpoint)
            if train_config.use_tpu:
                def tpu_scaffold():
                    tf.compat.v1.train.init_from_checkpoint(train_config.init_checkpoint, assignment_map)
                    return tf.compat.v1.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.compat.v1.tain.init_from_checkpoint(train_config.init_checkpoint, assignment_map)

        tf_logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf_logging.info("name = %s, shape = %s%s", var.name, var.shape, init_string)

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
                    "loss": total_loss,
            }
            output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
                    mode=mode,
                    loss=total_loss,
                    predictions=predictions,
                    scaffold_fn=scaffold_fn)


        return output_spec

    return model_fn

def input_fn_builder_masked(input_files, flags, is_training, num_cpu_threads=4):
    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]
        all_features = {}
        all_features.update(get_lm_basic_features(flags))
        all_features.update(get_lm_mask_features(flags))

        active_feature = ["input_ids", "input_mask", "segment_ids",
                          "next_sentence_labels",
                          "masked_lm_positions", "masked_lm_ids", "masked_lm_weights"
                          ]
        selected_features = {k: all_features[k] for k in active_feature}
        FixedLenFeature = tf.io.FixedLenFeature
        max_predictions_per_seq = flags.max_predictions_per_seq
        selected_features["loss_base"] = FixedLenFeature([max_predictions_per_seq], tf.float32)
        selected_features["loss_target"] = FixedLenFeature([max_predictions_per_seq], tf.float32)
        return format_dataset(selected_features, batch_size, is_training, flags, input_files, num_cpu_threads)
    return input_fn

