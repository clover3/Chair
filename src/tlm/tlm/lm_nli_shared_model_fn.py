import tensorflow as tf

from data_generator.special_tokens import MASK_ID
from models.transformer import optimization_v2 as optimization
from models.transformer.bert_common_v2 import get_shape_list2, dropout, create_attention_mask_from_input_mask2
from tf_util.tf_logging import tf_logging
from tlm.model import base, get_hidden_v2
from tlm.model.base import BertModel, mimic_pooling
from tlm.model.lm_objective import get_masked_lm_output
from tlm.model.masking import random_masking
from tlm.model.units import ForwardLayer
from tlm.training.assignment_map import get_bert_assignment_map
from tlm.training.input_fn_common import format_dataset
from tlm.training.lm_model_fn import metric_fn_lm
from tlm.training.model_fn_common import log_features, get_init_fn, get_tpu_scaffold_or_init, log_var_assignments, \
    classification_metric_fn, Classification


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


def shared_gradient_inner(vars, loss, y2):
    grads_1 = tf.gradients(ys=loss, xs=vars)
    grads_2 = tf.gradients(ys=y2, xs=vars)
    acc = 0
    for g1, g2 in zip(grads_1, grads_2):
        if g1 is not None and g2 is not None:
            acc += tf.reduce_sum(tf.abs(g1 * g2))
    return acc


def shared_gradient(loss, y2):
    return shared_gradient_inner(tf.compat.v1.trainable_variables(), loss, y2)


def shared_gradient_fine_grained(losses, y2, n_predictions):
    tvars = tf.compat.v1.trainable_variables()
    grads_2 = tf.gradients(ys=y2, xs=tvars)

    l = []
    for i in range(n_predictions):
        def inner(loss):
            grads_1 = tf.gradients(ys=loss, xs=tvars)

            acc = 0
            for g1, g2 in zip(grads_1, grads_2):
                if g1 is not None and g2 is not None:
                    acc += tf.reduce_sum(tf.abs(g1 * g2))
            return acc
        l.append(inner(losses[i]))

    return tf.stack(l)


class SimpleSharingModel:
    def __init__(self, config, use_one_hot_embeddings, is_training,
                 masked_input_ids, input_mask, segment_ids,
                 nli_input_ids, nli_input_mask, nli_segment_ids,
                 ):

        all_input_ids = tf.concat([masked_input_ids, nli_input_ids], axis=0)
        all_input_mask = tf.concat([input_mask, nli_input_mask], axis=0)
        all_segment_ids = tf.concat([segment_ids, nli_segment_ids], axis=0)
        self.batch_size, _ = get_shape_list2(masked_input_ids)
        self.model = BertModel(
            config,
            is_training,
            all_input_ids,
            all_input_mask,
            all_segment_ids,
            use_one_hot_embeddings
        )

    def lm_sequence_output(self):
        return self.model.get_sequence_output()[:self.batch_size]

    def get_embedding_table(self):
        return self.model.get_embedding_table()

    def get_tt_feature(self):
        return self.model.get_pooled_output()[self.batch_size:]


class SharingFetchGradModel:
    def __init__(self, config, use_one_hot_embeddings, is_training,
                 masked_input_ids, input_mask, segment_ids,
                 nli_input_ids, nli_input_mask, nli_segment_ids,
                 ):

        all_input_ids = tf.concat([masked_input_ids, nli_input_ids], axis=0)
        all_input_mask = tf.concat([input_mask, nli_input_mask], axis=0)
        all_segment_ids = tf.concat([segment_ids, nli_segment_ids], axis=0)
        self.batch_size, _ = get_shape_list2(masked_input_ids)
        self.model = get_hidden_v2.BertModel(
            config,
            is_training,
            all_input_ids,
            all_input_mask,
            all_segment_ids,
            use_one_hot_embeddings
        )

    def lm_sequence_output(self):
        return self.model.get_sequence_output()[:self.batch_size]

    def get_lm_hidden_layers(self):
        layers = []
        for l in self.model.all_layer_outputs:
            layers.append(l)
        return layers

    def get_embedding_table(self):
        return self.model.get_embedding_table()

    def get_tt_feature(self):
        return self.model.get_pooled_output()[self.batch_size:]

class AddLayerSharingModel:
    def __init__(self, config, use_one_hot_embeddings, is_training,
                 masked_input_ids, input_mask, segment_ids,
                 tt_input_ids, tt_input_mask, tt_segment_ids,
                 ):

        all_input_ids = tf.concat([masked_input_ids, tt_input_ids], axis=0)
        all_input_mask = tf.concat([input_mask, tt_input_mask], axis=0)
        all_segment_ids = tf.concat([segment_ids, tt_segment_ids], axis=0)
        self.config = config
        self.lm_batch_size, _ = get_shape_list2(masked_input_ids)
        self.model = BertModel(
            config,
            is_training,
            all_input_ids,
            all_input_mask,
            all_segment_ids,
            use_one_hot_embeddings
        )
        initializer = base.create_initializer(config.initializer_range)
        self.tt_layer = ForwardLayer(config, initializer)
        
        self.tt_input_mask = tt_input_mask
        seq_output = self.model.get_sequence_output()[self.lm_batch_size:]
        tt_batch_size, seq_length = get_shape_list2(tt_input_ids)
        tt_attention_mask = create_attention_mask_from_input_mask2(
            seq_output, self.tt_input_mask)

        print('tt_attention_mask', tt_attention_mask.shape)
        print("seq_output", seq_output.shape)
        seq_output = self.tt_layer.apply_3d(seq_output, tt_batch_size, seq_length, tt_attention_mask)
        self.tt_feature = mimic_pooling(seq_output, self.config.hidden_size, self.config.initializer_range)

    def lm_sequence_output(self):
        return self.model.get_sequence_output()[:self.lm_batch_size]

    def get_embedding_table(self):
        return self.model.get_embedding_table()

    def get_tt_feature(self):
        return self.tt_feature

def const_combine(loss1, loss2):
    loss = loss1 + loss2 * 0.1
    return loss

def decay_combine(loss1, loss2):
    combine_factor_init = 0.1
    num_warmup_steps = 10000
    num_train_steps= 1000 * 1000
    combine_factor = tf.constant(value=combine_factor_init, shape=[], dtype=tf.float32)
    global_step = tf.compat.v1.train.get_or_create_global_step()

    # Implements linear decay of the learning rate.
    combine_factor = tf.compat.v1.train.polynomial_decay(
        combine_factor,
        global_step,
        num_train_steps,
        end_learning_rate=0.0,
        power=1.0,
        cycle=False)

    global_steps_int = tf.cast(global_step, tf.int32)
    warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)

    global_steps_float = tf.cast(global_steps_int, tf.float32)
    warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

    warmup_percent_done = global_steps_float / warmup_steps_float
    warmup_learning_rate = combine_factor_init * warmup_percent_done

    is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
    combine_factor = (
            (1.0 - is_warmup) * combine_factor + is_warmup * warmup_learning_rate)

    loss = loss1 + loss2 * combine_factor
    return loss


def model_fn_nli_lm(config, train_config, sharing_model_factory, combine_loss_fn=const_combine):
    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        tf_logging.info("model_fn_nli_lm")
        """The `model_fn` for TPUEstimator."""
        log_features(features)

        input_ids = features["input_ids"] # [batch_size, seq_length]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        batch_size, _ = get_shape_list2(input_ids)
        if "nli_input_ids" in features:
            nli_input_ids = features["nli_input_ids"] # [batch_size, seq_length]
            nli_input_mask = features["nli_input_mask"]
            nli_segment_ids = features["nli_segment_ids"]
        else:
            nli_input_ids = input_ids
            nli_input_mask = input_mask
            nli_segment_ids = segment_ids
            features["label_ids"] = tf.ones([batch_size], tf.int32)

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

        sharing_model = sharing_model_factory(
            config,
            train_config.use_one_hot_embeddings,
            is_training,
            masked_input_ids,
            input_mask,
            segment_ids,
            nli_input_ids,
            nli_input_mask,
            nli_segment_ids
        )

        sequence_output_lm = sharing_model.lm_sequence_output()
        nli_feature = sharing_model.get_tt_feature()

        masked_lm_loss, masked_lm_example_loss, masked_lm_log_probs \
            = get_masked_lm_output(config, sequence_output_lm, sharing_model.get_embedding_table(),
                                     masked_lm_positions, masked_lm_ids, masked_lm_weights)

        masked_lm_log_probs = tf.reshape(masked_lm_log_probs, [batch_size, -1])

        top_guess = masked_lm_log_probs

        task = Classification(3, features, nli_feature, is_training)
        nli_loss = task.loss

        overlap_score = shared_gradient_fine_grained(masked_lm_example_loss, task.logits, train_config.max_predictions_per_seq )
        loss = combine_loss_fn(masked_lm_loss, nli_loss)
        tvars = tf.compat.v1.trainable_variables()
        assignment_fn = get_bert_assignment_map
        initialized_variable_names, init_fn = get_init_fn(tvars, train_config.init_checkpoint, assignment_fn)
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
                    "masked_lm_log_probs": masked_lm_log_probs,
                    "overlap_score": overlap_score,
                    "top_guess": top_guess,
            }
            output_spec = TPUEstimatorSpec(
                    mode=mode,
                    predictions=predictions,
                    scaffold_fn=scaffold_fn)

        return output_spec
    return model_fn



def model_fn_share_fetch_grad(config, train_config, sharing_model_factory, combine_loss_fn=const_combine):
    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        tf_logging.info("model_fn_nli_lm")
        """The `model_fn` for TPUEstimator."""
        log_features(features)

        input_ids = features["input_ids"] # [batch_size, seq_length]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        batch_size, seq_max = get_shape_list2(input_ids)
        if "nli_input_ids" in features:
            nli_input_ids = features["nli_input_ids"] # [batch_size, seq_length]
            nli_input_mask = features["nli_input_mask"]
            nli_segment_ids = features["nli_segment_ids"]
        else:
            nli_input_ids = input_ids
            nli_input_mask = input_mask
            nli_segment_ids = segment_ids
            features["label_ids"] = tf.ones([batch_size], tf.int32)

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

        sharing_model = sharing_model_factory(
            config,
            train_config.use_one_hot_embeddings,
            is_training,
            masked_input_ids,
            input_mask,
            segment_ids,
            nli_input_ids,
            nli_input_mask,
            nli_segment_ids
        )

        sequence_output_lm = sharing_model.lm_sequence_output()
        nli_feature = sharing_model.get_tt_feature()

        masked_lm_loss, masked_lm_example_loss, masked_lm_log_probs \
            = get_masked_lm_output(config, sequence_output_lm, sharing_model.get_embedding_table(),
                                     masked_lm_positions, masked_lm_ids, masked_lm_weights)

        masked_lm_log_probs = tf.reshape(masked_lm_log_probs, [batch_size, -1])

        masked_lm_per_inst_loss = tf.reshape(masked_lm_example_loss, [batch_size, -1])



        task = Classification(3, features, nli_feature, is_training)
        nli_loss = task.loss

        task_prob = tf.nn.softmax(task.logits, axis=-1)
        arg_like = task_prob[:, 1] + task_prob[:, 2]

        vars = sharing_model.model.all_layer_outputs
        grads_1 = tf.gradients(ys=masked_lm_loss, xs=vars) # List[ batch_szie,
        grads_2 = tf.gradients(ys=arg_like, xs=vars)
        l = []
        for g1, g2 in zip(grads_1, grads_2):
            if g1 is not None and g2 is not None:
                a = tf.reshape(g1, [batch_size*2, seq_max, -1]) [:batch_size]
                a = a / masked_lm_per_inst_loss
                b = tf.reshape(g2, [batch_size * 2, seq_max, -1])[batch_size:]
                l.append(tf.abs(a * b))
        h_overlap = tf.stack(l, axis=1)
        h_overlap = tf.reduce_sum(h_overlap, axis=2)

        loss = combine_loss_fn(masked_lm_loss, nli_loss)
        tvars = tf.compat.v1.trainable_variables()
        assignment_fn = get_bert_assignment_map
        initialized_variable_names, init_fn = get_init_fn(tvars, train_config.init_checkpoint, assignment_fn)
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
                    "masked_lm_log_probs": masked_lm_log_probs,
                    "h_overlap":h_overlap,
            }
            output_spec = TPUEstimatorSpec(
                    mode=mode,
                    predictions=predictions,
                    scaffold_fn=scaffold_fn)

        return output_spec
    return model_fn


def input_fn_builder(input_files, flags, is_training, use_next_sentence_labels, num_cpu_threads=4):

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
        if use_next_sentence_labels:
            all_features["next_sentence_labels"] = FixedLenFeature([1], tf.int64)

        return format_dataset(all_features, batch_size, is_training, flags, input_files, num_cpu_threads)

    return input_fn



