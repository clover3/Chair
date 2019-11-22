import re

import tensorflow as tf

from data_generator.special_tokens import MASK_ID
from models.transformer import optimization_v2 as optimization
from tlm.model.lm_objective import get_masked_lm_output, get_next_sentence_output
from tlm.model.masking import random_masking
from tlm.training.input_fn import _decode_record
from tlm.training.model_fn import metric_fn, get_assignment_map_as_is
from trainer.get_param_num import get_param_num
from tlm.training.grad_accumulation import get_accumulated_optimizer_from_config
from models.transformer import bert_common_v2 as bert_common

class DictRunConfig:
    def __init__(self,
                 use_target_pos_emb=False,
                 is_bert_checkpoint=True,
                 train_op="LM",
                 prediction_op="",
                 ):
        self.use_target_pos_emb = use_target_pos_emb
        self.is_bert_checkpoint = is_bert_checkpoint
        self.train_op = train_op
        self.prediction_op = prediction_op
    @classmethod
    def from_flags(cls, flags):
        return DictRunConfig(
            flags.use_target_pos_emb,
            flags.is_bert_checkpoint,
            flags.train_op,
            flags.prediction_op
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


def sequence_index_prediction(bert_config, gold_idx, input_tensor):
    logits = bert_common.dense(2, bert_common.create_initializer(bert_config.initializer_range))(input_tensor)
    log_probs = tf.nn.softmax(logits, axis=2)
    one_hot = tf.one_hot(gold_idx, bert_config.max_position_embeddings)
    losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, one_hot)
    per_example_loss = tf.reduce_sum(losses, axis=1)
    loss = tf.reduce_mean(per_example_loss)

    return loss, per_example_loss, log_probs


def model_fn_dict_reader(bert_config, train_config, logging, model_class, dict_run_config):
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

        if dict_run_config.prediction_op == "loss_fixed_mask":
            masked_input_ids = input_ids
            masked_lm_positions = features["masked_lm_positions"]
            masked_lm_ids = features["masked_lm_ids"]
            masked_lm_weights = tf.ones_like(masked_lm_positions, dtype=tf.float32)
        else:
            masked_input_ids, masked_lm_positions, masked_lm_ids, masked_lm_weights \
                = random_masking(input_ids, input_mask, train_config.max_predictions_per_seq, MASK_ID, seed)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        model = model_class(
                config=bert_config,
                d_config=bert_config,
                is_training=is_training,
                input_ids=masked_input_ids,
                input_mask=input_mask,
                d_input_ids=d_input_ids,
                d_input_mask=d_input_mask,
                d_location_ids=d_location_ids,
                use_target_pos_emb=dict_run_config.use_target_pos_emb,
                token_type_ids=segment_ids,
                use_one_hot_embeddings=train_config.use_one_hot_embeddings,
        )

        (masked_lm_loss,
         masked_lm_example_loss, masked_lm_log_probs) = get_masked_lm_output(
                 bert_config, model.get_sequence_output(), model.get_embedding_table(),
                 masked_lm_positions, masked_lm_ids, masked_lm_weights)
        (next_sentence_loss, next_sentence_example_loss,
         next_sentence_log_probs) = get_next_sentence_output(
                 bert_config, model.get_pooled_output(), next_sentence_labels)

        if dict_run_config.train_op == "lookup":
            lookup_idx = features["lookup_idx"]
            lookup_loss, lookup_example_loss, lookup_score = \
                sequence_index_prediction(bert_config, lookup_idx, model.get_sequence_output())

        total_loss = masked_lm_loss

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
                        "input_ids": input_ids,
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
        FixedLenFeature = tf.io.FixedLenFeature
        all_features = {
            "input_ids":    FixedLenFeature([max_seq_length], tf.int64),
            "input_mask":   FixedLenFeature([max_seq_length], tf.int64),
            "segment_ids":  FixedLenFeature([max_seq_length], tf.int64),
            "d_input_ids":  FixedLenFeature([max_def_length], tf.int64),
            "d_input_mask": FixedLenFeature([max_def_length], tf.int64),
            "d_location_ids": FixedLenFeature([max_loc_length], tf.int64),
            "next_sentence_labels": FixedLenFeature([1], tf.int64),
            "masked_lm_positions": FixedLenFeature([max_predictions_per_seq], tf.int64),
            "masked_lm_ids": FixedLenFeature([max_predictions_per_seq], tf.int64),
            "lookup_idx": FixedLenFeature([1], tf.int64),
        }

        active_feature = ["input_ids", "input_mask", "segment_ids",
                          "d_input_ids", "d_input_mask", "d_location_ids",
                          "next_sentence_label"]

        if flags.fixed_mask:
                active_feature.append("masked_lm_positions")
                active_feature.append("masked_lm_ids")
        if flags.train_op == "lookup":
                active_feature.append("lookup_idx")

        name_to_features = {k:all_features[k] for k in active_feature}

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        if is_training:
            d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
            d = d.repeat()
            d = d.shuffle(buffer_size=1000* 1000)

            # `cycle_length` is the number of parallel files that get read.
            cycle_length = min(num_cpu_threads, len(input_files))
            cycle_length = 100
            # `sloppy` mode means that the interleaving is not exact. This adds
            # even more randomness to the training pipeline.
            d = d.apply(
                    tf.data.experimental.parallel_interleave(
                            tf.data.TFRecordDataset,
                            sloppy=is_training,
                            cycle_length=cycle_length))
            d = d.shuffle(buffer_size=100)
        else:
            d = tf.data.TFRecordDataset(input_files)
            n_predict = flags.eval_batch_size * flags.max_pred_steps
            d = d.take(n_predict)

            # Since we evaluate for a fixed number of steps we don't want to encounter
            # out-of-range exceptions.
            #d = d.repeat()

        # We must `drop_remainder` on training because the TPU requires fixed
        # size dimensions. For eval, we assume we are evaluating on the CPU or GPU
        # and we *don't* want to drop the remainder, otherwise we wont cover
        # every sample.
        d = d.apply(
                tf.data.experimental.map_and_batch(
                        lambda record: _decode_record(record, name_to_features),
                        batch_size=batch_size,
                        num_parallel_batches=num_cpu_threads,
                        drop_remainder=True))
        return d

    return input_fn