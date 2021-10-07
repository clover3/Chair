import tensorflow as tf

from explain.pairing.common_unit import probe_modeling
from explain.pairing.match_predictor import ProbeConfigI
from models.transformer.transformer_cls import transformer_pooled
from tf_util.tf_logging import tf_logging
from trainer.multi_gpu_support import get_multiple_models, get_avg_loss, get_avg_tensors_from_models, \
    get_batch2feed_dict_for_multi_gpu, get_concat_tensors_from_models, \
    get_concat_tensors_list_from_models, get_train_op
from trainer.tf_train_module import get_train_op2


# class ProbeClassifier : probe_model.py : network-wise modeling
# common_util.py : probe_modeling.py : Per layer modeling


class ProbeSet:
    def __init__(self,
                 main_model: transformer_pooled,
                 hp,
                 per_layer_component: str,
                 use_embedding_out: bool
                 ):
        all_layers = main_model.model.get_all_encoder_layers()  # List[ tensor[batch,seq_length, hidden] ]
        if use_embedding_out:
            all_layers = [main_model.model.get_embedding_output()] + all_layers
        num_labels = hp.num_classes
        self.main_model = main_model
        _, input_mask, _ = main_model.x_list
        probe_target = tf.expand_dims(tf.nn.softmax(main_model.logits, axis=-1), axis=1)
        probe_target = tf.tile(probe_target, [1, hp.seq_max, 1])
        if per_layer_component == 'linear':
            def network(layer_output_fixed):
                logits = tf.layers.dense(layer_output_fixed, num_labels)
                return logits
        elif per_layer_component == 'mlp':
            def network(layer_output_fixed):
                hidden = tf.layers.dense(layer_output_fixed, 768, activation='relu')
                logits = tf.layers.dense(hidden, num_labels)
                return logits
        else:
            assert False

        with tf.variable_scope("match_predictor"):
            per_layer_models = list([probe_modeling(layer, probe_target, input_mask, network)
                                     for layer_no, layer in enumerate(all_layers)])
        self.per_layer_models = per_layer_models

        loss = 0
        for d in per_layer_models:
            loss += d.loss

        self.all_losses = tf.stack([d.loss for d in per_layer_models])
        self.loss = loss
        self.per_layer_logits = list([d.logits for d in per_layer_models])


def build_model(hp, lms_model_config: ProbeConfigI):
    main_model = transformer_pooled(hp, hp.vocab_size)
    probe_set = ProbeSet(main_model,
                         hp,
                         lms_model_config.per_layer_component,
                         lms_model_config.use_embedding_out
                         )
    return main_model, probe_set


class ClsProbeModel:
    def __init__(self, bert_hp, probe_config, num_gpu):
        def build_model_fn():
            return build_model(bert_hp, probe_config)

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