import tensorflow as tf

from models.transformer.optimization_v2 import create_optimizer_from_config
from tf_util.tf_logging import tf_logging
from tlm.model.base import BertModel
from tlm.model_cnfig import JsonConfig
from tlm.qtype.qtype_model_fn import set_dropout_to_zero, dummy_fn
from tlm.training.assignment_map import get_init_fn
from tlm.training.model_fn_common import log_features, log_var_assignments, get_tpu_scaffold_or_init, \
    classification_metric_fn
from tlm.training.train_config import TrainConfigEx


def qtype_prediction_model_fn(FLAGS):
    model_config_o = JsonConfig.from_json_file(FLAGS.model_config_file)
    train_config = TrainConfigEx.from_flags(FLAGS)
    use_one_hot_embeddings = FLAGS.use_tpu
    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        log_features(features)
        input_ids = features["entity"]
        input_mask_f = tf.equal(input_ids, 0)
        input_mask = tf.cast(input_mask_f, tf.int64)
        segment_ids = tf.zeros_like(input_ids)
        label_ids = features["qtype_id"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        if not is_training:
            model_config = set_dropout_to_zero(model_config_o)
        else:
            model_config = model_config_o

        model = BertModel(
            config=model_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings,
        )
        pooled = model.get_pooled_output() # [batch_size * 2, hidden_size]

        n_type = model_config.q_voca_size
        logits = tf.keras.layers.Dense(n_type)(pooled)
        probs = tf.nn.softmax(logits, axis=1)

        y_true = tf.one_hot(label_ids, n_type)
        y_true = tf.reshape(y_true, [-1, n_type])
        loss_arr = tf.keras.losses.categorical_crossentropy(y_true, logits, from_logits=True,)
        loss = tf.reduce_mean(loss_arr)
        prediction = {
            "data_id": features["data_id"],
            "label_ids": features["qtype_id"],
            "logits": logits
        }
        all_tvars = tf.compat.v1.trainable_variables()
        if train_config.init_checkpoint:
            initialized_variable_names, init_fn = get_init_fn(train_config, all_tvars)
            log_var_assignments(all_tvars, initialized_variable_names)
        else:
            init_fn = dummy_fn
        scaffold_fn = get_tpu_scaffold_or_init(init_fn, train_config.use_tpu)
        optimizer_factory = lambda x: create_optimizer_from_config(x, train_config)
        TPUEstimatorSpec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec
        if mode == tf.estimator.ModeKeys.TRAIN:
            tf_logging.info("Using single lr ")
            train_op = optimizer_factory(loss)
            output_spec = TPUEstimatorSpec(mode=mode, loss=loss, train_op=train_op, scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:
            eval_metrics = (classification_metric_fn, [
                probs, label_ids, tf.ones_like(probs)
            ])
            output_spec = TPUEstimatorSpec(mode=mode, loss=loss, eval_metrics=eval_metrics,
                                           scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.PREDICT:
            output_spec = TPUEstimatorSpec(mode=mode, loss=loss,
                                           predictions=prediction,
                                           scaffold_fn=scaffold_fn)
        else:
            assert False
        return output_spec


    return model_fn