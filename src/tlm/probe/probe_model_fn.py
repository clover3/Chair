import tensorflow as tf
from keras.utils import losses_utils

from explain.pairing.common_unit import probe_modeling, probability_mae
from models.transformer.optimization_v2 import create_optimizer_from_config
from tf_util.tf_logging import tf_logging
from tlm.model.base import BertModelInterface, BertModel
from tlm.model_cnfig import JsonConfig
from tlm.qtype.qtype_model_fn import set_dropout_to_zero
from tlm.training.assignment_map import get_init_fn
from tlm.training.model_fn_common import log_features, get_tpu_scaffold_or_init
from tlm.training.train_config import TrainConfigEx


def kl_divergence(true_y, logits):
    pred_prob = tf.nn.softmax(logits, axis=-1)
    return tf.keras.losses.KLDivergence(losses_utils.ReductionV2.NONE)(true_y, pred_prob)


class ProbeSet:
    def __init__(self,
                 main_model: BertModelInterface,
                 input_mask,
                 logits,
                 num_labels,
                 max_seq_length,
                 modeling: str,
                 use_embedding_out: bool
                 ):
        all_layers = main_model.get_all_encoder_layers()  # List[ tensor[batch,seq_length, hidden] ]
        if use_embedding_out:
            all_layers = [main_model.get_embedding_output()] + all_layers
        self.main_model = main_model
        probe_target = tf.expand_dims(tf.nn.softmax(logits, axis=-1), axis=1)
        probe_target = tf.tile(probe_target, [1, max_seq_length, 1])
        Dense = tf.keras.layers.Dense
        per_layer_component, loss_opt = modeling.split("_")
        if per_layer_component == 'linear':
            def network(layer_output_fixed):
                logits = Dense(num_labels)(layer_output_fixed)
                return logits
        elif per_layer_component == 'mlp':
            def network(layer_output_fixed):
                hidden = Dense(768, activation='relu')(layer_output_fixed)
                logits = Dense(num_labels)(hidden)
                return logits
        else:
            assert False

        if loss_opt == "pmae":
            loss_fn = probability_mae
        elif loss_opt == "kl":
            loss_fn = kl_divergence
            print("using kl")
        else:
            assert False


        with tf.compat.v1.variable_scope("match_predictor"):
            per_layer_models = list([probe_modeling(layer, probe_target, input_mask, network, loss_fn)
                                     for layer_no, layer in enumerate(all_layers)])
        self.per_layer_models = per_layer_models

        loss = 0
        for d in per_layer_models:
            loss += d.loss

        self.all_losses = tf.stack([d.loss for d in per_layer_models])
        self.loss = loss
        self.per_layer_logits = list([d.logits for d in per_layer_models])



def find_padding(input_mask):
    return tf.where(input_mask == 0)[0][0]


def append_prefix_to_dict_key(d, prefix):
    return {prefix + "_" + k: v for k, v in d.items()}


def get_metric_fn(num_layers, max_seq_length):
    def metric_fn(losses, per_layer_logits, cls_logits, input_ids, input_mask, segment_ids):
        """Computes the loss and accuracy of the model."""
        gold_pred = tf.argmax(cls_logits, axis=1)
        gold_pred_exd = tf.tile(tf.expand_dims(gold_pred, axis=1), [1, max_seq_length])

        def get_layer_metrics(layer_logits):
            def get_metric_with_mask(weights):
                pred = tf.argmax(
                    input=layer_logits, axis=2, output_type=tf.int32)
                accuracy = tf.compat.v1.metrics.accuracy(
                    labels=gold_pred_exd, predictions=pred, weights=weights)

                precision = tf.compat.v1.metrics.precision(
                    labels=gold_pred_exd, predictions=pred, weights=weights)

                recall = tf.compat.v1.metrics.recall(
                    labels=gold_pred_exd, predictions=pred, weights=weights)
                return {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                }
            is_cls = tf.equal(input_ids, 101)
            is_seg1 = tf.logical_and(tf.equal(segment_ids, 0), tf.cast(input_mask, tf.bool))
            todo = [
                (is_seg1, "seg1"),
                (segment_ids, "seg2"),
                (input_mask, "non_padding"),
                (is_cls, 'CLS')
            ]
            d_per_layer = {}
            for mask, name in todo:
                d = get_metric_with_mask(mask)
                new_d = append_prefix_to_dict_key(d, name)
                d_per_layer.update(new_d)
            return d_per_layer

        output = {}
        for i in range(num_layers):
            d = get_layer_metrics(per_layer_logits[:, i])
            new_d = append_prefix_to_dict_key(d, "layer {}".format(i))
            output.update(new_d)
            output["Layer {} loss".format(i)] = tf.compat.v1.metrics.mean(losses[i])

        return output
    return metric_fn


def model_fn_probe(FLAGS):
    model_config_o = JsonConfig.from_json_file(FLAGS.model_config_file)
    train_config = TrainConfigEx.from_flags(FLAGS)

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        log_features(features)
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        if not is_training:
            model_config = set_dropout_to_zero(model_config_o)
        else:
            model_config = model_config_o

        model_config_predict = set_dropout_to_zero(model_config_o)
        model = BertModel(
            config=model_config_predict,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=train_config.use_one_hot_embeddings,
        )
        pooled_output = model.get_pooled_output()
        logits = tf.keras.layers.Dense(train_config.num_classes, name="cls_dense")(pooled_output)
        probe_set = ProbeSet(model, input_mask, logits, train_config.num_classes,
                             FLAGS.max_seq_length, FLAGS.modeling, True)
        loss = probe_set.loss
        all_losses = probe_set.all_losses
        tvars = tf.compat.v1.trainable_variables()
        initialized_variable_names, init_fn = get_init_fn(train_config, tvars)
        tf_logging.info("There are {} trainable variables".format(len(tvars)))
        scaffold_fn = get_tpu_scaffold_or_init(init_fn, train_config.use_tpu)
        optimizer_factory = lambda x: create_optimizer_from_config(x, train_config, tvars)
        TPUEstimatorSpec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec
        per_layer_logit_tensors = tf.stack(probe_set.per_layer_logits, axis=1)
        metric_fn = get_metric_fn(13, FLAGS.max_seq_length)
        if mode == tf.estimator.ModeKeys.TRAIN:
            tf_logging.info("Using single lr ")
            train_op = optimizer_factory(loss)
            output_spec = TPUEstimatorSpec(mode=mode, loss=loss, train_op=train_op, scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:
            eval_metrics = (metric_fn, [
                all_losses, per_layer_logit_tensors, logits, input_ids, input_mask, segment_ids
            ])
            output_spec = TPUEstimatorSpec(mode=mode, loss=loss, eval_metrics=eval_metrics,
                                           scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.PREDICT:
            prediction = {
                'per_layer_logits': per_layer_logit_tensors,
                'logits': logits,
                'input_ids': input_ids
            }
            output_spec = TPUEstimatorSpec(mode=mode, loss=loss,
                                           predictions=prediction,
                                           scaffold_fn=scaffold_fn)
        else:
            assert False
        return output_spec

    return model_fn
