import tensorflow as tf

from explain.pairing.probe.model_core import build_model
from tf_util.tf_logging import tf_logging
from trainer.multi_gpu_support import get_multiple_models, get_avg_loss, get_avg_tensors_from_models, \
    get_batch2feed_dict_for_multi_gpu, get_concat_tensors_from_models, \
    get_concat_tensors_list_from_models, get_train_op
from trainer.tf_train_module import get_train_op2


# class ProbeClassifier : probe_model.py : network-wise modeling
# common_util.py : probe_modeling.py : Per layer modeling


class ClsProbeModel:
    def __init__(self, bert_hp, probe_config, num_gpu, is_training):
        def build_model_fn():
            return build_model(bert_hp, probe_config, is_training)

        self.num_gpu = num_gpu
        self.match_predictor = None
        self.prob_predictor_list = None
        self.bert_hp = bert_hp

        if num_gpu == 1:
            tf_logging.info("Using single GPU")
            task_model_, probe_predictor_list = build_model_fn()
            loss_tensor = probe_predictor_list.loss
            per_layer_loss = probe_predictor_list.all_losses
            batch2feed_dict = task_model_.batch2feed_dict
            logits = task_model_.logits
            per_layer_logit_tensor = probe_predictor_list.per_layer_logits
            self.match_predictor = probe_predictor_list
        else:
            main_models, probe_predictor_list = zip(*get_multiple_models(build_model_fn, num_gpu))
            loss_tensor = get_avg_loss(probe_predictor_list)
            per_layer_loss = get_avg_tensors_from_models(probe_predictor_list,
                                                         lambda match_predictor: match_predictor.all_losses)
            batch2feed_dict = get_batch2feed_dict_for_multi_gpu(main_models)
            logits = get_concat_tensors_from_models(main_models, lambda model: model.logits)

            per_layer_logit_tensor = \
                get_concat_tensors_list_from_models(probe_predictor_list, lambda model: model.per_layer_logits)

            self.prob_predictor_list = probe_predictor_list

        self.logits = logits
        self.batch2feed_dict = batch2feed_dict
        self.loss_tensor = loss_tensor
        self.per_layer_logit_tensor = per_layer_logit_tensor
        self.per_layer_loss = per_layer_loss

    # logits for nli classification
    def get_logits(self):
        return self.logits

    # logits for match score of each layers
    def get_lms(self):
        return self.per_layer_logit_tensor

    def get_train_op(self, lr, max_steps):
        scope_name = "probe_optimizer"
        if self.num_gpu == 1:
            with tf.variable_scope(scope_name):
                train_op = get_train_op2(self.match_predictor.loss, lr, "adam", max_steps)
        else:
            with tf.variable_scope(scope_name):
                train_op = get_train_op([m.loss for m in self.prob_predictor_list], lr, max_steps)

        return train_op