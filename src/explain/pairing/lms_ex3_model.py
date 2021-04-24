import tensorflow as tf

from explain.explain_model import CrossEntropyModeling, CorrelationModeling
from explain.pairing.ex3way_predictor import build_model
from tf_util.tf_logging import tf_logging
from trainer.multi_gpu_support import get_multiple_models, get_avg_loss, get_avg_tensors_from_models, \
    get_batch2feed_dict_for_multi_gpu, get_concat_tensors_from_models, get_concat_tensors_list_from_models, get_train_op
from trainer.tf_train_module import get_train_op2


class LMSModel:
    def __init__(self, modeling_option, bert_hp, lms_config, num_gpu):
        ex_modeling_class = {
            'ce': CrossEntropyModeling,
            'co': CorrelationModeling
        }[modeling_option]

        def build_model_fn():
            return build_model(ex_modeling_class, bert_hp, lms_config)

        self.num_gpu = num_gpu
        self.match_predictor = None
        self.match_predictor_list = None
        self.bert_hp = bert_hp

        if num_gpu == 1:
            tf_logging.info("Using single GPU")
            task_model_, ex_model_, match_predictor_3way = build_model_fn()
            loss_tensor = match_predictor_3way.loss
            # List[Tensor[num_layer]]
            per_layer_loss_list = match_predictor_3way.all_losses
            batch2feed_dict = task_model_.batch2feed_dict
            logits = task_model_.logits
            ex_score_tensor = ex_model_.get_scores()

            # List[List[Tensor[batch, 2]]]
            per_layer_logit_list = match_predictor_3way.per_layer_logits_list

            self.match_predictor = match_predictor_3way

        else:
            main_models, ex_models, match_predictor_list = zip(*get_multiple_models(build_model_fn, num_gpu))
            loss_tensor = get_avg_loss(match_predictor_list)
            per_layer_loss_list = [
                get_avg_tensors_from_models(match_predictor_list,
                                            lambda match_predictor: match_predictor.per_layer_loss_list[i])
                                            for i in range(lms_config.num_tags)
            ]
            batch2feed_dict = get_batch2feed_dict_for_multi_gpu(main_models)
            logits = get_concat_tensors_from_models(main_models, lambda model: model.logits)

            def get_loss_tensor(model):
                t = tf.expand_dims(tf.stack(model.get_losses()), 0)
                return t

            ex_score_tensor = get_concat_tensors_from_models(ex_models,
                                                             lambda model: model.get_scores())
            per_layer_logit_list = [
                get_concat_tensors_list_from_models(match_predictor_list, lambda model: model.per_layer_logit_list[i])
                for i in range(lms_config.num_tags)
            ]

            self.match_predictor_list = match_predictor_list

        self.logits = logits
        self.batch2feed_dict = batch2feed_dict
        self.ex_score_tensor = ex_score_tensor
        self.loss_tensor = loss_tensor

        # List[List[Tensor[batch_size, 2]]]
        self.per_layer_logit_list = per_layer_logit_list
        # List[List[Tensor[1]]]
        self.per_layer_loss_list = per_layer_loss_list

    # logits for nli classification
    def get_logits(self):
        return self.logits

    # logits for match score of each layers
    def get_lms(self):
        return self.per_layer_logit_list

    def get_train_op(self, lr, max_steps):
        if self.num_gpu == 1:
            with tf.variable_scope("match_optimizer"):
                train_op = get_train_op2(self.match_predictor.loss, lr, "adam", max_steps)
        else:
            with tf.variable_scope("match_optimizer"):
                train_op = get_train_op([m.loss for m in self.match_predictor_list], lr, max_steps)

        return train_op