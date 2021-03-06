import re

import tensorflow as tf

from data_generator.special_tokens import MASK_ID
from models.transformer import bert_common_v2 as bert_common
from models.transformer import optimization_v2 as optimization
from tlm.model.lm_objective import get_masked_lm_output, get_next_sentence_output
from tlm.model.masking import random_masking
from tlm.training.assignment_map import get_assignment_map_as_is
from tlm.training.grad_accumulation import get_accumulated_optimizer_from_config
from tlm.training.input_fn_common import format_dataset
from tlm.training.lm_model_fn import metric_fn
from trainer.get_param_num import get_param_num


class DictRunConfig:
    def __init__(self,
                 use_target_pos_emb=False,
                 is_bert_checkpoint=True,
                 train_op="LM",
                 prediction_op="",
                 pool_dict_output=False,
                 inner_batch_size= None,
                 def_per_batch=None,
                 use_d_segment_ids=False,
                 use_ab_mapping_mask=False,
                 max_def_length=None,
                 ):
        self.use_target_pos_emb = use_target_pos_emb
        self.is_bert_checkpoint = is_bert_checkpoint
        self.train_op = train_op
        self.prediction_op = prediction_op
        self.pool_dict_output = pool_dict_output
        self.inner_batch_size = inner_batch_size
        self.def_per_batch = def_per_batch
        self.use_d_segment_ids = use_d_segment_ids
        self.use_ab_mapping_mask = use_ab_mapping_mask
        self.max_def_length = max_def_length

    @classmethod
    def from_flags(cls, flags):
        return DictRunConfig(
            flags.use_target_pos_emb,
            flags.is_bert_checkpoint,
            flags.train_op,
            flags.prediction_op,
            flags.pool_dict_output,
            flags.inner_batch_size,
            flags.def_per_batch,
            flags.use_d_segment_ids,
            flags.use_ab_mapping_mask,
            flags.max_def_length,
        )


def get_bert_assignment_map_for_dict(tvars, lm_checkpoint):
    candidate_1 = {}
    candidate_2 = {}
    real_name_map_1 = {}
    real_name_map_2 = {}

    remap_scope = "dict"

    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)

        name_in_checkpoint = re.sub("layer_normalization[_]?\d*", "LayerNorm", name)
        name_in_checkpoint = re.sub("dense[_]?\d*", "dense", name_in_checkpoint)
        name_in_checkpoint = re.sub("encoder[_]?\d*", "encoder", name_in_checkpoint)

        f_dict_var = False
        tokens = name_in_checkpoint.split("/")
        for idx, t in enumerate(tokens):
            if t == remap_scope or t.startswith(remap_scope+"_"):
                name_in_checkpoint = "/".join(tokens[:idx] + tokens[idx+1:])
                f_dict_var = True
        if not f_dict_var:
            candidate_1[name_in_checkpoint] = var
            real_name_map_1[name_in_checkpoint] = name
        else:
            candidate_2[name_in_checkpoint] = var
            real_name_map_2[name_in_checkpoint] = name

    assignment_map_1 = {}
    assignment_map_2 = {}
    initialized_variable_names = {}
    if lm_checkpoint:
        for x in tf.train.list_variables(lm_checkpoint):
            (name, var) = (x[0], x[1])
            if name not in candidate_1:
                (name, var) = (x[0], x[1])
                continue
            assignment_map_1[name] = candidate_1[name]
            tvar_name = real_name_map_1[name]
            initialized_variable_names[tvar_name] = 1
            initialized_variable_names[tvar_name + ":0"] = 1

        for x in tf.train.list_variables(lm_checkpoint):
            (name, var) = (x[0], x[1])
            if name not in candidate_2:
                continue
            assignment_map_2[name] = candidate_2[name]
            tvar_name = real_name_map_2[name]
            initialized_variable_names[tvar_name] = 1
            initialized_variable_names[tvar_name + ":0"] = 1

    return assignment_map_1, assignment_map_2, initialized_variable_names


def get_gradients(model, masked_lm_log_probs, n_pred, voca_size):
    masked_lm_log_probs = tf.reshape(masked_lm_log_probs, [-1, n_pred, voca_size])

    g_list = []
    for i in range(n_pred):
        g = tf.gradients(masked_lm_log_probs[:,i,:], model.d_embedding_output)
        g_list.append(g)
    return tf.stack(g_list)




def sequence_index_prediction(bert_config, lookup_idx, input_tensor):
    logits = bert_common.dense(2, bert_common.create_initializer(bert_config.initializer_range))(input_tensor)
    log_probs = tf.nn.softmax(logits, axis=2)
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=lookup_idx)
    per_example_loss = tf.reduce_sum(losses, axis=1)
    loss = tf.reduce_mean(per_example_loss)

    return loss, per_example_loss, log_probs


def binary_prediction(bert_config, input_tensor):
    logits = bert_common.dense(2, bert_common.create_initializer(bert_config.initializer_range))(input_tensor)
    log_probs = tf.nn.softmax(logits, axis=2)
    return logits, log_probs


def model_fn_dict_reader(bert_config, dbert_config, train_config, logging, model_class, dict_run_config):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):    # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""
        logging.info("*** Features ***")
        for name in sorted(features.keys()):
            logging.info("    name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        d_input_ids = features["d_input_ids"]
        d_input_mask = features["d_input_mask"]
        d_location_ids = features["d_location_ids"]
        next_sentence_labels = features["next_sentence_labels"]

        if dict_run_config.prediction_op == "loss":
            seed = 0
        else:
            seed = None

        if dict_run_config.prediction_op == "loss_fixed_mask" or train_config.fixed_mask:
            masked_input_ids = input_ids
            masked_lm_positions = features["masked_lm_positions"]
            masked_lm_ids = features["masked_lm_ids"]
            masked_lm_weights = tf.ones_like(masked_lm_positions, dtype=tf.float32)
        else:
            masked_input_ids, masked_lm_positions, masked_lm_ids, masked_lm_weights \
                = random_masking(input_ids, input_mask, train_config.max_predictions_per_seq, MASK_ID, seed)

        if dict_run_config.use_d_segment_ids:
            d_segment_ids = features["d_segment_ids"]
        else:
            d_segment_ids = None

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        model = model_class(
                config=bert_config,
                d_config=dbert_config,
                is_training=is_training,
                input_ids=masked_input_ids,
                input_mask=input_mask,
                d_input_ids=d_input_ids,
                d_input_mask=d_input_mask,
                d_location_ids=d_location_ids,
                use_target_pos_emb=dict_run_config.use_target_pos_emb,
                token_type_ids=segment_ids,
                use_one_hot_embeddings=train_config.use_one_hot_embeddings,
                d_segment_ids=d_segment_ids,
                pool_dict_output=dict_run_config.pool_dict_output,
        )

        (masked_lm_loss,
         masked_lm_example_loss, masked_lm_log_probs) = get_masked_lm_output(
                 bert_config, model.get_sequence_output(), model.get_embedding_table(),
                 masked_lm_positions, masked_lm_ids, masked_lm_weights)
        (next_sentence_loss, next_sentence_example_loss,
         next_sentence_log_probs) = get_next_sentence_output(
                 bert_config, model.get_pooled_output(), next_sentence_labels)

        total_loss = masked_lm_loss

        if dict_run_config.train_op == "entry_prediction":
            score_label = features["useful_entry"] # [batch, 1]
            score_label = tf.reshape(score_label, [-1])
            entry_logits = bert_common.dense(2, bert_common.create_initializer(bert_config.initializer_range))\
                (model.get_dict_pooled_output())
            print("entry_logits: ", entry_logits.shape)
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=entry_logits, labels=score_label)
            loss = tf.reduce_mean(losses)
            total_loss = loss

        if dict_run_config.train_op == "lookup":
            lookup_idx = features["lookup_idx"]
            lookup_loss, lookup_example_loss, lookup_score = \
                sequence_index_prediction(bert_config, lookup_idx, model.get_sequence_output())

            total_loss += lookup_loss

        tvars = tf.compat.v1.trainable_variables()

        init_vars = {}
        scaffold_fn = None
        if train_config.init_checkpoint:
            if dict_run_config.is_bert_checkpoint:
                map1, map2, init_vars = get_bert_assignment_map_for_dict(tvars, train_config.init_checkpoint)

                def load_fn():
                    tf.compat.v1.train.init_from_checkpoint(train_config.init_checkpoint, map1)
                    tf.compat.v1.train.init_from_checkpoint(train_config.init_checkpoint, map2)
            else:
                map1, init_vars = get_assignment_map_as_is(tvars, train_config.init_checkpoint)

                def load_fn():
                    tf.compat.v1.train.init_from_checkpoint(train_config.init_checkpoint, map1)

            if train_config.use_tpu:
                def tpu_scaffold():
                    load_fn()
                    return tf.compat.v1.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                load_fn()

        logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in init_vars:
                init_string = ", *INIT_FROM_CKPT*"
            logging.info("    name = %s, shape = %s%s", var.name, var.shape, init_string)
        logging.info("Total parameters : %d" % get_param_num())

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            if train_config.gradient_accumulation == 1:
                train_op = optimization.create_optimizer_from_config(total_loss, train_config)
            else:
                logging.info("Using gradient accumulation : %d" % train_config.gradient_accumulation)
                train_op = get_accumulated_optimizer_from_config(total_loss, train_config,
                                                                 tvars, train_config.gradient_accumulation)
            output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
                    mode=mode,
                    loss=total_loss,
                    train_op=train_op,
                    scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:
            eval_metrics = (metric_fn, [
                    masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
                    masked_lm_weights, next_sentence_example_loss,
                    next_sentence_log_probs, next_sentence_labels
            ])
            output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
                    mode=mode,
                    loss=total_loss,
                    eval_metrics=eval_metrics,
                    scaffold_fn=scaffold_fn)
        else:
            if dict_run_config.prediction_op == "gradient":
                logging.info("Fetching gradient")
                gradient = get_gradients(model, masked_lm_log_probs,
                                         train_config.max_predictions_per_seq, bert_config.vocab_size)
                predictions = {
                        "masked_input_ids": masked_input_ids,
                        #"input_ids": input_ids,
                        "d_input_ids": d_input_ids,
                        "masked_lm_positions": masked_lm_positions,
                        "gradients": gradient,
                }
            elif dict_run_config.prediction_op == "loss" or dict_run_config.prediction_op == "loss_fixed_mask":
                logging.info("Fetching loss")
                predictions = {
                    "masked_lm_example_loss": masked_lm_example_loss,
                }
            else:
                raise Exception("prediction target not specified")

            output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
                    mode=mode,
                    loss=total_loss,
                    predictions=predictions,
                    scaffold_fn=scaffold_fn)

        return output_spec

    return model_fn


def input_fn_builder_dict(input_files, flags, is_training, num_cpu_threads=4):
    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]
        max_seq_length = flags.max_seq_length
        max_predictions_per_seq = flags.max_predictions_per_seq
        max_def_length = flags.max_def_length
        max_loc_length = flags.max_loc_length
        max_word_length = flags.max_word_length
        FixedLenFeature = tf.io.FixedLenFeature
        all_features = {
            "input_ids":    FixedLenFeature([max_seq_length], tf.int64),
            "input_mask":   FixedLenFeature([max_seq_length], tf.int64),
            "segment_ids":  FixedLenFeature([max_seq_length], tf.int64),
            "d_input_ids":  FixedLenFeature([max_def_length], tf.int64),
            "d_input_mask": FixedLenFeature([max_def_length], tf.int64),
            "d_segment_ids": FixedLenFeature([max_def_length], tf.int64),
            "d_location_ids": FixedLenFeature([max_loc_length], tf.int64),
            "next_sentence_labels": FixedLenFeature([1], tf.int64),
            "masked_lm_positions": FixedLenFeature([max_predictions_per_seq], tf.int64),
            "masked_lm_ids": FixedLenFeature([max_predictions_per_seq], tf.int64),
            "lookup_idx": FixedLenFeature([1], tf.int64),
            "selected_word": FixedLenFeature([max_word_length], tf.int64),
        }

        active_feature = ["input_ids", "input_mask", "segment_ids",
                          "d_input_ids", "d_input_mask", "d_location_ids",
                          "next_sentence_labels"]

        if flags.fixed_mask:
            active_feature.append("masked_lm_positions")
            active_feature.append("masked_lm_ids")
        if flags.train_op == "lookup":
            active_feature.append("masked_lm_positions")
            active_feature.append("masked_lm_ids")
            active_feature.append("lookup_idx")
        elif flags.train_op == "entry_prediction":
            active_feature.append("masked_lm_positions")
            active_feature.append("masked_lm_ids")
            active_feature.append("lookup_idx")

        if flags.use_d_segment_ids:
            active_feature.append("d_segment_ids")

        if max_word_length > 0:
            active_feature.append("selected_word")

        selected_features = {k:all_features[k] for k in active_feature}

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        return format_dataset(selected_features, batch_size, is_training, flags, input_files, num_cpu_threads)

    return input_fn


