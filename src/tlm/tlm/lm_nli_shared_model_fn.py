import tensorflow as tf

from data_generator.special_tokens import MASK_ID
from models.transformer import optimization_v2 as optimization
from models.transformer.bert_common_v2 import get_shape_list2, dropout
from tf_util.tf_logging import tf_logging
from tlm.model.base import BertModel
from tlm.model.lm_objective import get_masked_lm_output
from tlm.model.masking import random_masking
from tlm.training.assignment_map import get_bert_assignment_map
from tlm.training.input_fn_common import format_dataset
from tlm.training.lm_model_fn import metric_fn_lm
from tlm.training.model_fn_common import log_features, align_checkpoint, get_tpu_scaffold_or_init, log_var_assignments, \
    classification_metric_fn


class ClassificationIgnore12:
    def __init__(self, num_classes, features, rep, is_training):
        self.num_classes = num_classes
        self.label_ids = features["label_ids"]
        self.label_ids = tf.cast(tf.equal(self.label_ids, 0), tf.int32)
        self.label_ids = tf.reshape(self.label_ids, [-1])
        if "is_real_example" in features:
            is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
        else:
            is_real_example = tf.ones(tf.shape(self.label_ids), dtype=tf.float32)
        self.is_real_example = is_real_example

        if is_training:
            rep = dropout(rep, 0.1)
        logits = tf.keras.layers.Dense(self.num_classes, name="cls_dense")(rep)

        self.logits = logits
        self.loss_arr = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits,
            labels=self.label_ids)
        self.loss = tf.reduce_mean(input_tensor=self.loss_arr)
        self.preds = tf.cast(tf.argmax(logits, axis=-1), dtype=tf.int32)

    def eval_metrics(self):
        eval_metrics = (classification_metric_fn, [
            self.logits, self.label_ids, self.is_real_example
        ])
        return eval_metrics

def model_fn_nli_lm(config, train_config):
    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        tf_logging.info("model_fn_apr_lm")
        """The `model_fn` for TPUEstimator."""
        log_features(features)

        input_ids = features["input_ids"] # [batch_size, seq_length]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]

        nli_input_ids = features["nli_input_ids"] # [batch_size, seq_length]
        nli_input_mask = features["nli_input_mask"]
        nli_segment_ids = features["nli_segment_ids"]

        batch_size, _ = get_shape_list2(input_ids)

        if mode == tf.estimator.ModeKeys.PREDICT:
            tf.random.set_seed(0)
            seed = 0
        else:
            seed = None

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        tf_logging.info("Doing dynamic masking (random)")

        masked_input_ids, masked_lm_positions, masked_lm_ids, masked_lm_weights \
            = random_masking(input_ids, input_mask,
                             train_config.max_predictions_per_seq, MASK_ID, seed)

        all_input_ids = tf.concat([masked_input_ids, nli_input_ids], axis=0)
        all_input_mask = tf.concat([input_mask, nli_input_mask], axis=0)
        all_segment_ids = tf.concat([segment_ids, nli_segment_ids], axis=0)

        model_class = BertModel

        model = model_class(
            config,
            is_training,
            all_input_ids,
            all_input_mask,
            all_segment_ids,
            train_config.use_one_hot_embeddings
        )

        sequence_output_lm = model.get_sequence_output()[:batch_size]

        masked_lm_loss, masked_lm_example_loss, masked_lm_log_probs \
            = get_masked_lm_output(config, sequence_output_lm, model.get_embedding_table(),
                                     masked_lm_positions, masked_lm_ids, masked_lm_weights)

        pooled = model.get_pooled_output()[batch_size:]
        #task = Classification(3, features, pooled, is_training)
        task = ClassificationIgnore12(3, features, pooled, is_training)
        nli_loss = task.loss
        loss = masked_lm_loss + nli_loss
        tvars = tf.compat.v1.trainable_variables()
        assignment_fn = get_bert_assignment_map
        initialized_variable_names, init_fn = align_checkpoint(tvars, train_config.init_checkpoint, assignment_fn)
        scaffold_fn = get_tpu_scaffold_or_init(init_fn, train_config.use_tpu)
        log_var_assignments(tvars, initialized_variable_names)

        TPUEstimatorSpec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec
        if mode == tf.estimator.ModeKeys.TRAIN:
            train_op = optimization.create_optimizer_from_config(loss, train_config)
            output_spec = TPUEstimatorSpec(
                    mode=mode,
                    loss=loss,
                    train_op=train_op,
                    scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:
            eval_metrics = (metric_fn_lm, [
                    masked_lm_example_loss,
                    masked_lm_log_probs,
                    masked_lm_ids,
                    masked_lm_weights,
            ])
            output_spec = TPUEstimatorSpec(
                    mode=mode,
                    loss=loss,
                    eval_metrics=eval_metrics,
                    scaffold_fn=scaffold_fn)
        else:
            predictions = {
                    "input_ids": input_ids,
                    "masked_input_ids": masked_input_ids,
                    "masked_lm_ids": masked_lm_ids,
                    "masked_lm_example_loss": masked_lm_example_loss,
                    "masked_lm_positions": masked_lm_positions,
            }
            output_spec = TPUEstimatorSpec(
                    mode=mode,
                    loss=loss,
                    predictions=predictions,
                    scaffold_fn=scaffold_fn)

        return output_spec
    return model_fn


def input_fn_builder(input_files, flags, is_training, num_cpu_threads=4):

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]
        max_sequence_length = flags.max_seq_length
        FixedLenFeature = tf.io.FixedLenFeature
        all_features = {
            "input_ids":    FixedLenFeature([max_sequence_length], tf.int64),
            "input_mask":   FixedLenFeature([max_sequence_length], tf.int64),
            "segment_ids":  FixedLenFeature([max_sequence_length], tf.int64),
            "nli_input_ids": FixedLenFeature([max_sequence_length], tf.int64),
            "nli_input_mask": FixedLenFeature([max_sequence_length], tf.int64),
            "nli_segment_ids": FixedLenFeature([max_sequence_length], tf.int64),
            "label_ids": FixedLenFeature([1], tf.int64),
        }
        return format_dataset(all_features, batch_size, is_training, flags, input_files, num_cpu_threads)

    return input_fn
