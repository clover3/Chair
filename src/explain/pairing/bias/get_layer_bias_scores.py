import models.bert_util.bert_utils
from explain.pairing.match_predictor import build_model, LMSConfig
from tf_util.tf_logging import tf_logging


def lms_predict(bert_hp,
                train_config, lms_config: LMSConfig, save_dir, nli_data, modeling_option, init_fn):


    def build_model_fn():
        return build_model(ex_modeling_class, bert_hp, lms_config)

    tf_logging.info("Using single GPU")
    task_model_, ex_model_, match_predictor = build_model_fn()
    loss_tensor = match_predictor.loss
    per_layer_loss = match_predictor.all_losses
    with tf.variable_scope("match_optimizer"):
        train_cls = get_train_op2(match_predictor.loss, bert_hp.lr, "adam", max_steps)
    batch2feed_dict = models.bert_util.bert_utils.batch2feed_dict_4_or_5_inputs
    logits = task_model_.logits
    ex_score_tensor = ex_model_.get_ex_scores(lms_config.target_idx)
    per_layer_logit_tensor = match_predictor.per_layer_logits

    pass