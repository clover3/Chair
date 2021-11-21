import tensorflow as tf

from models.transformer.bert_common_v2 import create_initializer
from models.transformer.optimization_v2 import create_optimizer_from_config
from tf_util.tf_logging import tf_logging
from tlm.model.base import BertModel
from tlm.model_cnfig import JsonConfig
from tlm.qtype.qde_model_fn import get_mask_from_input_ids, InputTriplet
from tlm.qtype.qtype_model_fn import set_dropout_to_zero, dummy_fn
from tlm.training.assignment_map import get_init_fn
from tlm.training.model_fn_common import log_features, log_var_assignments, get_tpu_scaffold_or_init
from tlm.training.train_config import TrainConfigEx


def process_feature(features):
    de_inputs = InputTriplet(input_ids=features["d_e_input_ids"],
                             segment_ids=features["d_e_segment_ids"],
                             input_mask=get_mask_from_input_ids(features["d_e_input_ids"])
                             )
    qtype_id = features["qtype_id"]
    return qtype_id, de_inputs


def fixed_qtype_metric(y_pred):
    """Computes the loss and accuracy of the model."""
    is_correct = tf.cast(tf.less(0.0, y_pred), tf.int32)
    acc = tf.compat.v1.metrics.accuracy(tf.ones_like(is_correct), is_correct)
    return {
        "pairwise_acc": acc,
    }


def qtype_id_modeling_identity(model_config, qtype_id):
    n_qtype = model_config.q_voca_size
    valid_mask = tf.cast(tf.less(qtype_id, n_qtype), tf.int32)
    masked_qtype_id = qtype_id * valid_mask

    return tf.reshape(tf.one_hot(masked_qtype_id, n_qtype), [-1, n_qtype])


def model_fn_fixed_qtype(FLAGS,
                  process_feature,
                  get_loss_modeling,
                  get_qtype_id_modeling,
                  get_qtype_modeling_de,
                  ):
    def single_bias_model(config, vector):
        dense = tf.keras.layers.Dense(1, activation=tf.keras.activations.tanh,
                                      kernel_initializer=create_initializer(config.initializer_range))
        v = dense(vector)
        return tf.reshape(v, [-1])

    model_config_o = JsonConfig.from_json_file(FLAGS.model_config_file)
    train_config = TrainConfigEx.from_flags(FLAGS)
    use_one_hot_embeddings = FLAGS.use_tpu
    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        log_features(features)
        qtype_id, de_inputs = process_feature(features)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        if not is_training:
            model_config = set_dropout_to_zero(model_config_o)
        else:
            model_config = model_config_o

        qtype_vector1 = get_qtype_id_modeling(model_config, qtype_id)
        model = BertModel(
            config=model_config,
            is_training=is_training,
            input_ids=de_inputs.input_ids,
            input_mask=de_inputs.input_mask,
            token_type_ids=de_inputs.segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings,
        )
        pooled2 = model.get_pooled_output()
        qtype_vector2 = get_qtype_modeling_de(model_config, pooled2)
        d_bias = single_bias_model(model_config, pooled2)
        print(qtype_vector1)
        print(qtype_vector2)
        query_document_score = tf.reduce_sum(tf.multiply(qtype_vector1, qtype_vector2), axis=1)
        print(query_document_score)
        bias = tf.Variable(initial_value=0.0, trainable=True)
        query_document_score += bias
        query_document_score += d_bias
        try:

            alpha = model_config.alpha
        except Exception as e:
            print(e)
            alpha = 1
        tf_logging.info("Using alpha of {}".format(alpha))

        loss, losses, y_pred = get_loss_modeling(query_document_score, features)
        prediction = {
            "data_id": features["data_id"],
            "label_ids": features["label_ids"],
            # "qe_input_ids": qe_input_ids,
            # "de_input_ids": de_input_ids,
            # "qtype_vector_qe": qtype_vector1,
            # "qtype_vector_de": qtype_vector2,
            "logits": query_document_score,
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
            eval_metrics = (fixed_qtype_metric, [
                y_pred,
            ])
            output_spec = TPUEstimatorSpec(mode=mode, loss=loss, eval_metrics=eval_metrics,
                                           scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.PREDICT:
            if prediction is None:
                prediction = {
                    "y_pred": y_pred,
                    "losses": losses,
                }
            output_spec = TPUEstimatorSpec(mode=mode, loss=loss,
                                           predictions=prediction,
                                           scaffold_fn=scaffold_fn)
        else:
            assert False
        return output_spec


    return model_fn
