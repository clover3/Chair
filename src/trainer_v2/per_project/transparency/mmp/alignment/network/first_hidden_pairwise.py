from trainer_v2.custom_loop.modeling_common.network_utils import TwoLayerDense
from trainer_v2.per_project.transparency.mmp.alignment.network.align_net_v2 import TFBertLayerFlat
from trainer_v2.per_project.transparency.mmp.alignment.network.align_net_v3 import build_probe_from_layer_features
from trainer_v2.per_project.transparency.mmp.alignment.network.common import mean_pool_over_masked, \
    get_emb_concat_feature, build_align_acc_dict_pairwise, define_pairwise_input_and_stack_flat, \
    form_input_ids_segment_ids, define_pairwise_galign_inputs
from typing import Dict, Any
import tensorflow as tf
from transformers import shape_list, BertConfig, TFBertForSequenceClassification

from trainer_v2.per_project.transparency.mmp.probe.probe_common import get_attn_mask_bias, identify_layers
from trainer_v2.per_project.transparency.mmp.trnsfmr_util import get_dummy_input_for_bert_layer


def extract_features_from_first_hidden(bert_main_layer, bert_flat, input_ids, token_type_ids, q_term_mask):
    embedding_output = bert_main_layer.embeddings(
        input_ids=input_ids,
        token_type_ids=token_type_ids,
        training=False,
    )
    input_mask = tf.cast(tf.not_equal(input_ids, 0), tf.int32)
    input_mask_shape = shape_list(input_mask)
    attn_mask = tf.reshape(
        input_mask, (input_mask_shape[0], 1, 1, input_mask_shape[1])
    )
    dtype = embedding_output.dtype
    attn_mask_bias = get_attn_mask_bias(attn_mask, dtype)

    def pool_by_q_term_mask(vector):
        return mean_pool_over_masked(vector, q_term_mask)

    per_layer_feature_tensors = bert_flat(embedding_output, attn_mask_bias)
    return input_mask, per_layer_feature_tensors, pool_by_q_term_mask


class GAlignFirstHiddenPairwise:
    def __init__(self, tokenizer):
        n_out_dim = 1
        target_layer_no = 0
        cls_id = tokenizer.vocab["[CLS]"]
        sep_id = tokenizer.vocab["[SEP]"]

        bert_config = BertConfig()
        bert_cls = TFBertForSequenceClassification(bert_config)
        bert_main_layer = bert_cls.bert
        _ = bert_main_layer(get_dummy_input_for_bert_layer())

        # Define layers
        layers_d = identify_layers(bert_main_layer, target_layer_no)
        bert_flat = TFBertLayerFlat(bert_config, layers_d)

        # Build inputs
        max_term_len = 1
        q_term_flat, d_term_flat, inputs = define_pairwise_input_and_stack_flat(max_term_len)
        d_term_mask, input_ids, q_term_mask, token_type_ids = form_input_ids_segment_ids(
            q_term_flat, d_term_flat,
            cls_id, sep_id,
            max_term_len)

        # Extract features
        input_mask, per_layer_feature_tensors, pool_by_q_term_mask = extract_features_from_first_hidden(
            bert_main_layer, bert_flat, input_ids, token_type_ids, q_term_mask)

        def projection_layer(probe_name):
            hidden_size = bert_config.hidden_size
            return TwoLayerDense(hidden_size, n_out_dim, name=probe_name, activation2=None)

        align_probe = build_probe_from_layer_features(
            per_layer_feature_tensors, bert_config.hidden_size, projection_layer, pool_by_q_term_mask)

        emb_concat_probe = get_emb_concat_feature(bert_main_layer, projection_layer, q_term_flat, d_term_flat)
        align_probe['emb_concat'] = emb_concat_probe
        for k, v in align_probe.items():
            print(k, v)

        align_probe['align_pred'] = align_probe['g_attention_output']

        output_d = {
            "align_probe": align_probe,
            "input_mask": input_mask,
            "q_term_mask": q_term_mask,
            "d_term_mask": d_term_mask,
        }

        self.probe_model_output: Dict[str, Any] = output_d
        self.model = tf.keras.models.Model(
            inputs=inputs, outputs=self.probe_model_output)

    def get_align_metrics(self) -> Dict[str, tf.keras.metrics.Metric]:
        d = self.probe_model_output["align_probe"]
        output_d = build_align_acc_dict_pairwise(d)
        return output_d



class GAlignFirstHiddenTwoModel:
    def __init__(self, tokenizer, classifier_layer_builder):
        bert_config = BertConfig()
        bert_config.num_labels = 1
        bert_cls = TFBertForSequenceClassification(bert_config)
        bert_main_layer = bert_cls.bert
        _ = bert_main_layer(get_dummy_input_for_bert_layer())
        project = tf.keras.layers.Dense(1)
        cls_id = tokenizer.vocab["[CLS]"]
        sep_id = tokenizer.vocab["[SEP]"]

        max_term_len = 1
        q_term_flat, d_term_flat, inputs = define_pairwise_input_and_stack_flat(max_term_len)

        d_term_mask, input_ids, q_term_mask, token_type_ids = form_input_ids_segment_ids(
            q_term_flat, d_term_flat,
            cls_id, sep_id,
            max_term_len)

        bert_output = bert_main_layer(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            training=False,
            return_dict=True,
        )
        features = bert_output.pooler_output
        align_pred = project(features)

        align_probe = {
            'align_pred': align_pred
        }
        output_d = {
            "align_probe": align_probe,
        }

        self.probe_model_output: Dict[str, Any] = output_d
        self.pair_model = tf.keras.models.Model(
            inputs=inputs, outputs=self.probe_model_output)

        q_term = tf.keras.layers.Input(shape=(max_term_len,), dtype='int32', name="q_term")
        d_term = tf.keras.layers.Input(shape=(max_term_len,), dtype='int32', name="d_term")
        inputs = [q_term, d_term]

        d_term_mask, input_ids, q_term_mask, token_type_ids = form_input_ids_segment_ids(
            q_term, d_term,
            cls_id, sep_id,
            max_term_len)

        bert_output = bert_main_layer(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            training=False,
            return_dict=True,
        )
        features = bert_output.pooler_output
        align_pred = project(features)
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
