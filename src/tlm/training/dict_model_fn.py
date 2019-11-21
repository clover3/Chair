import re

import tensorflow as tf

from data_generator.special_tokens import MASK_ID
from models.transformer import optimization_v2 as optimization
from tlm.model.lm_objective import get_masked_lm_output, get_next_sentence_output
from tlm.model.masking import random_masking
from tlm.training.model_fn import metric_fn
from trainer.get_param_num import get_param_num
from tlm.training.grad_accumulation import get_accumulated_optimizer_from_config


def get_dict_bert_assignment_map(tvars, lm_checkpoint):
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


def model_fn_dict_reader(bert_config, train_config, logging, model_class, prediction_op=""):
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

        if prediction_op == "loss":
            seed = 0
        else:
            seed = None

        if prediction_op == "loss_fixed_mask":
            masked_input_ids = input_ids
            masked_lm_positions = features["masked_lm_positions"]
            masked_lm_ids = features["masked_lm_labels"]
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
                use_target_pos_emb=train_config.use_target_pos_emb,
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

        total_loss = masked_lm_loss

        tvars = tf.compat.v1.trainable_variables()

        initialized_variable_names = {}
        scaffold_fn = None
        if train_config.init_checkpoint:
            (assignment_map1, assignment_map2, initialized_variable_names
            ) = get_dict_bert_assignment_map(tvars, train_config.init_checkpoint)

            def load_fn():
                tf.compat.v1.train.init_from_checkpoint(train_config.init_checkpoint, assignment_map1)
                tf.compat.v1.train.init_from_checkpoint(train_config.init_checkpoint, assignment_map2)

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
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            logging.info("    name = %s, shape = %s%s", var.name, var.shape,
                                            init_string)
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
            if prediction_op == "gradient":
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
            elif prediction_op == "loss":
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


