from typing import Dict, Any

import tensorflow as tf
from transformers import BertConfig, TFBertForSequenceClassification

from trainer_v2.per_project.transparency.mmp.alignment.network.align_net5 import \
    apply_to_bert_main_layer_stack_first_token_reps
from trainer_v2.per_project.transparency.mmp.alignment.network.common import build_align_acc_dict_pairwise, \
    define_pairwise_galign_inputs, define_pointwise_galign_inputs
from trainer_v2.per_project.transparency.mmp.trnsfmr_util import get_dummy_input_for_bert_layer


class GAlignClsStackPairwise:
    def __init__(self, tokenizer, classifier_layer_builder):
        bert_config = BertConfig()
        bert_config.num_labels = 1
        bert_cls = TFBertForSequenceClassification(bert_config)
        bert_main_layer = bert_cls.bert
        _ = bert_main_layer(get_dummy_input_for_bert_layer())

        max_term_len = 1
        input_ids, token_type_ids, inputs = \
            define_pairwise_galign_inputs(max_term_len, tokenizer)
        feature_rep = apply_to_bert_main_layer_stack_first_token_reps(bert_main_layer, input_ids, token_type_ids)

        classifier = classifier_layer_builder()
        align_pred = classifier(feature_rep)
        align_probe = {
            'align_pred': align_pred
        }
        output_d = {
            "align_probe": align_probe,
        }

        self.probe_model_output: Dict[str, Any] = output_d
        self.model = tf.keras.models.Model(
            inputs=inputs, outputs=self.probe_model_output)

    def get_align_metrics(self) -> Dict[str, tf.keras.metrics.Metric]:
        d = self.probe_model_output["align_probe"]
        output_d = build_align_acc_dict_pairwise(d)
        return output_d



def tensor_list_hash(tensor_list):
    v_list = [tf.reduce_sum(t) for t in tensor_list]
    t_sum = tf.reduce_sum(v_list)
    return t_sum.numpy()


class GAlignClsStackTwoModel:
    def __init__(self, tokenizer, classifier_layer_builder):
        bert_config = BertConfig()
        bert_config.num_labels = 1
        bert_cls = TFBertForSequenceClassification(bert_config)
        bert_main_layer = bert_cls.bert
        _ = bert_main_layer(get_dummy_input_for_bert_layer())

        max_term_len = 1
        input_ids, token_type_ids, inputs = \
            define_pairwise_galign_inputs(max_term_len, tokenizer)

        feature_rep = apply_to_bert_main_layer_stack_first_token_reps(bert_main_layer, input_ids, token_type_ids)

        classifier = classifier_layer_builder()
        align_pred = classifier(feature_rep)
        align_probe = {
            'align_pred': align_pred
        }
        output_d = {
            "align_probe": align_probe,
        }

        self.probe_model_output: Dict[str, Any] = output_d
        self.pair_model = tf.keras.models.Model(
            inputs=inputs, outputs=self.probe_model_output)

        input_ids, token_type_ids, inputs = \
            define_pointwise_galign_inputs(max_term_len, tokenizer)
        feature_rep = apply_to_bert_main_layer_stack_first_token_reps(bert_main_layer, input_ids, token_type_ids)
        align_pred = classifier(feature_rep)
        align_probe = {
            'align_pred': align_pred
        }
        output_d = {
            "align_probe": align_probe,
        }

        self.probe_model_output_single: Dict[str, Any] = output_d
        self.single_model = tf.keras.models.Model(
            inputs=inputs, outputs=self.probe_model_output_single)

    def get_align_metrics(self) -> Dict[str, tf.keras.metrics.Metric]:
        d = self.probe_model_output["align_probe"]
        output_d = build_align_acc_dict_pairwise(d)
        return output_d

    def get_inference_model(self):
        return self.single_model

    def load_checkpoint(self, model_save_path):
        checkpoint = tf.train.Checkpoint(self.pair_model)
        checkpoint.restore(model_save_path).expect_partial()
