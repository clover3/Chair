import sys
from typing import List

import numpy as np
import tensorflow as tf

import data_generator
from cache import save_to_pickle
from data_generator.shared_setting import BertNLI
from explain.explain_model import CrossEntropyModeling, CorrelationModeling
from explain.pairing.match_predictor import MatchPredictor
from explain.pairing.run_train import HPCommon
from explain.pairing.train_pairing import NLIPairingTrainConfig
from explain.runner.nli_ex_param import ex_arg_parser
from explain.setups import init_fn_generic
from explain.train_nli import get_nli_data
from models.transformer.transformer_cls import transformer_pooled
from tf_util.tf_logging import tf_logging, set_level_debug, reset_root_log_handler
from trainer.multi_gpu_support import get_multiple_models, get_avg_loss, get_avg_tensors_from_models, \
    get_batch2feed_dict_for_multi_gpu, get_concat_tensors_from_models, get_concat_tensors_list_from_models
from trainer.tf_train_module import init_session


def predict_fn(sess, dev_batches,
                        loss_tensor, ex_scores_tensor, per_layer_logits_tensor,
                        batch2feed_dict):
    loss_list = []
    all_ex_scores: List[np.array] = []
    all_per_layer_logits: List[List[np.array]] = []
    input_ids_list = []
    input_mask_list = []
    segment_ids_list = []
    label_list = []
    for batch in dev_batches:
        input_ids, input_mask, segment_ids, label = batch
        input_ids_list.append(input_ids)
        input_mask_list.append(input_mask)
        segment_ids_list.append(segment_ids)
        label_list.append(label)
        loss_val, ex_scores, per_layer_logits \
            = sess.run([loss_tensor, ex_scores_tensor, per_layer_logits_tensor],
                       feed_dict=batch2feed_dict(batch)
                       )
        loss_list.append(loss_val)
        all_ex_scores.append(ex_scores)
        all_per_layer_logits.append(per_layer_logits)

    all_input_ids = np.concatenate(input_ids_list, axis=0)
    all_segment_ids = np.concatenate(segment_ids_list, axis=0)
    all_input_mask = np.concatenate(input_mask_list, axis=0)
    all_ex_scores_np = np.concatenate(all_ex_scores, axis=0)
    all_label = np.concatenate(label_list, axis=0)
    num_layer = len(per_layer_logits_tensor)
    logits_grouped_by_layer = []
    for layer_no in range(num_layer):
        t = np.concatenate([batch[layer_no] for batch in all_per_layer_logits], axis=0)
        logits_grouped_by_layer.append(t)

    output_d = {
        'input_ids': all_input_ids,
        "segment_ids": all_segment_ids,
        "input_mask": all_input_mask,
        "ex_scores": all_ex_scores_np,
        "label": all_label,
        "logits": logits_grouped_by_layer
    }
    return output_d


def do_predict(hparam, train_config, data,
                  tags, modeling_option, init_fn,
                  ):
    num_gpu = train_config.num_gpu
    train_batches, dev_batches = data

    ex_modeling_class = {
        'ce': CrossEntropyModeling,
        'co': CorrelationModeling
    }[modeling_option]

    def build_model():
        main_model = transformer_pooled(hparam, train_config.vocab_size)
        ex_model = ex_modeling_class(main_model.model.sequence_output, hparam.seq_max, len(tags),
                                     main_model.batch2feed_dict)
        match_predictor = MatchPredictor(main_model, ex_model, 2, "linear")
        return main_model, ex_model, match_predictor

    target_idx = 2
    if num_gpu == 1:
        tf_logging.info("Using single GPU")
        task_model_, ex_model_, match_predictor = build_model()
        loss_tensor = match_predictor.loss
        all_losses = match_predictor.all_losses
        batch2feed_dict = task_model_.batch2feed_dict
        logits = task_model_.logits
        ex_score_tensor = ex_model_.get_ex_scores(target_idx)
        per_layer_logit_tensor = match_predictor.per_layer_logits
    else:
        main_models, ex_models, match_predictors = zip(*get_multiple_models(build_model, num_gpu))
        loss_tensor = get_avg_loss(match_predictors)
        all_losses = get_avg_tensors_from_models(match_predictors, lambda match_predictor: match_predictor.all_losses)
        batch2feed_dict = get_batch2feed_dict_for_multi_gpu(main_models)
        logits = get_concat_tensors_from_models(main_models, lambda model: model.logits)
        def get_loss_tensor(model):
            t = tf.expand_dims(tf.stack(model.get_losses()), 0)
            return t
        ex_score_tensor = get_concat_tensors_from_models(ex_models, lambda model: model.get_ex_scores(target_idx))
        per_layer_logit_tensor = \
            get_concat_tensors_list_from_models(match_predictors, lambda model: model.per_layer_logits)

    sess = init_session()
    sess.run(tf.global_variables_initializer())
    init_fn(sess)

    # make explain train_op does not increase global step

    output_d = predict_fn(sess, dev_batches[:20], loss_tensor, ex_score_tensor,
                       per_layer_logit_tensor, batch2feed_dict)
    return output_d


def main(start_model_path, modeling_option, tags, num_gpu=1):
    num_gpu = int(num_gpu)
    hp = HPCommon()
    nli_setting = BertNLI()
    set_level_debug()
    reset_root_log_handler()
    train_config = NLIPairingTrainConfig()
    train_config.num_gpu = num_gpu

    def init_fn(sess):
        return init_fn_generic(sess, "as_is", start_model_path)
    data = get_nli_data(hp, nli_setting)


    output_d = do_predict(hp, train_config, data, tags, modeling_option, init_fn)
    save_to_pickle(output_d, "pairing_pred")


if __name__ == "__main__":
    args = ex_arg_parser.parse_args(sys.argv[1:])
    main(args.start_model_path,
               args.modeling_option,
               data_generator.NLI.nli_info.tags,
               args.num_gpu)
