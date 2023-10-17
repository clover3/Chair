from typing import Dict, Any

import tensorflow as tf
from transformers import shape_list, BertConfig, TFBertForSequenceClassification

from trainer_v2.custom_loop.modeling_common.network_utils import TwoLayerDense
from trainer_v2.per_project.transparency.mmp.alignment.network.align_net_v2 import TFBertLayerFlat
from trainer_v2.per_project.transparency.mmp.alignment.network.common import mean_pool_over_masked, \
    build_align_acc_dict_pairwise, define_pairwise_input_and_stack_flat, \
    reshape_per_head_features, build_input_ids_segment_ids
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


def apply_probe_layer_from_layer_features(all_hidden_var_d, projection_layer_d, seq_pooling_fn):
    probe_d = {}
    for k, feature_tensor in all_hidden_var_d.items():
        probe_name = k
        pooled = seq_pooling_fn(feature_tensor)
        projection = projection_layer_d[probe_name]
        probe_d[probe_name] = projection(pooled)
    return probe_d


def get_all_hidden_var_d(out_d, hidden_size):
    bmd_hidden_vars = [
        'layer_input_vector', 'attention_output',
        'g_attention_output', 'g_attention_output_add_residual',
        'intermediate_output', 'bert_out_last']
    bhmd_hidden_var_d = reshape_per_head_features(out_d, hidden_size)
    bmd_hidden_var_d = {k: out_d[k] for k in bmd_hidden_vars}
    hidden_vars = []
    hidden_vars.extend(bhmd_hidden_var_d.values())
    hidden_vars.extend(bmd_hidden_var_d.values())
    all_feature = tf.concat(hidden_vars, axis=2)
    arr = []
    arr.extend(bhmd_hidden_var_d.values())
    qkv_feature = tf.concat(arr, axis=2)
    concat_hidden_var_d = {
        "all_concat": all_feature,
        "qkv_feature": qkv_feature,
    }
    all_hidden_var_d = {}
    all_hidden_var_d.update(bhmd_hidden_var_d)
    all_hidden_var_d.update(bmd_hidden_var_d)
    all_hidden_var_d.update(concat_hidden_var_d)
    return all_hidden_var_d


def get_emb_concat_feature(bert_main_layer, q_term, d_term):
    q_term_emb = bert_main_layer.embeddings(input_ids=q_term)
    d_term_emb = bert_main_layer.embeddings(input_ids=d_term)
    t = tf.concat([q_term_emb, d_term_emb], axis=2)[:, 0, :]
    k = "emb_concat"
    t_stop = tf.stop_gradient(t, name=f"{k}_stop_gradient")
    return t_stop


class GAlignFirstHiddenPairwise:
    def __init__(self, tokenizer):
        n_out_dim = 1
        target_layer_no = 0

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
        d_term_mask, input_ids, q_term_mask, token_type_ids = build_input_ids_segment_ids(
            q_term_flat, d_term_flat,
            tokenizer,
            max_term_len)

        # Extract features
        input_mask, per_layer_feature_tensors, pool_by_q_term_mask = extract_features_from_first_hidden(
            bert_main_layer, bert_flat, input_ids, token_type_ids, q_term_mask)

        # Define probe layers
        def projection_layer_factory(probe_name):
            hidden_size = bert_config.hidden_size
            return TwoLayerDense(hidden_size, n_out_dim, name=probe_name, activation2=None)

        all_hidden_var_d = get_all_hidden_var_d(per_layer_feature_tensors, bert_config.hidden_size)

        all_probe_names = list(all_hidden_var_d.keys())
        probe_layer_d = {k: projection_layer_factory(k) for k in all_probe_names}
        align_probe = apply_probe_layer_from_layer_features(
            all_hidden_var_d, probe_layer_d, pool_by_q_term_mask)

        for k, v in align_probe.items():
            print(k, v)

        align_probe['align_pred'] = align_probe['g_attention_output']

        emb_concat_probe_layer = projection_layer_factory('emb_concat')
        emb_concat = get_emb_concat_feature(bert_main_layer, q_term_flat, d_term_flat)
        emb_concat_probe = emb_concat_probe_layer(emb_concat)
        align_probe['emb_concat'] = emb_concat_probe

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
    def __init__(self, tokenizer):
        n_out_dim = 1
        target_layer_no = 0

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
        d_term_mask, input_ids, q_term_mask, token_type_ids = build_input_ids_segment_ids(
            q_term_flat, d_term_flat, max_term_len, tokenizer)

        def extract_features_from_first_hidden_fn(input_ids, token_type_ids, q_term_mask):
            input_mask, per_layer_feature_tensors, pool_by_q_term_mask = extract_features_from_first_hidden(
                bert_main_layer, bert_flat, input_ids, token_type_ids, q_term_mask)
            return input_mask, per_layer_feature_tensors, pool_by_q_term_mask

        # Extract vectors from BERT
        input_mask, per_layer_feature_tensors, pool_by_q_term_mask = extract_features_from_first_hidden_fn(input_ids, token_type_ids, q_term_mask)

        # Define probe layers
        def projection_layer_factory(probe_name):
            hidden_size = bert_config.hidden_size
            return TwoLayerDense(hidden_size, n_out_dim, name=probe_name, activation2=None)

        all_hidden_var_d = get_all_hidden_var_d(per_layer_feature_tensors, bert_config.hidden_size)
        all_probe_names = list(all_hidden_var_d.keys())
        probe_layer_d = {k: projection_layer_factory(k) for k in all_probe_names}
        emb_concat_probe_layer = projection_layer_factory('emb_concat')

        ## Aply for pairwise model
        align_probe = apply_probe_layer_from_layer_features(
            all_hidden_var_d, probe_layer_d, pool_by_q_term_mask)
        emb_concat = get_emb_concat_feature(bert_main_layer, q_term_flat, d_term_flat)
        emb_concat_probe = emb_concat_probe_layer(emb_concat)
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
        self.pair_model = tf.keras.models.Model(
            inputs=inputs, outputs=self.probe_model_output)

        s_inputs, s_output_d = self.define_single_model(
            max_term_len, tokenizer,
            extract_features_from_first_hidden_fn, bert_config.hidden_size,
            probe_layer_d, bert_main_layer, emb_concat_probe_layer)

        self.probe_model_output_single: Dict[str, Any] = s_output_d
        self.single_model = tf.keras.models.Model(
            inputs=s_inputs, outputs=self.probe_model_output_single)

    def define_single_model(self, max_term_len, tokenizer,
                            extract_features_from_first_hidden_fn, hidden_size, probe_layer_d,
                            bert_main_layer, emb_concat_probe_layer):
        q_term = tf.keras.layers.Input(shape=(max_term_len,), dtype='int32', name="q_term")
        d_term = tf.keras.layers.Input(shape=(max_term_len,), dtype='int32', name="d_term")
        inputs = [q_term, d_term]

        d_term_mask, input_ids, q_term_mask, token_type_ids = build_input_ids_segment_ids(
            q_term, d_term,
            max_term_len, tokenizer)
        # Extract features
        input_mask, per_layer_feature_tensors, pool_by_q_term_mask = \
            extract_features_from_first_hidden_fn(input_ids, token_type_ids, q_term_mask)
        all_hidden_var_d = get_all_hidden_var_d(per_layer_feature_tensors, hidden_size)
        align_probe = apply_probe_layer_from_layer_features(
            all_hidden_var_d, probe_layer_d, pool_by_q_term_mask)
        emb_concat = get_emb_concat_feature(bert_main_layer, q_term, d_term)
        emb_concat_probe = emb_concat_probe_layer(emb_concat)
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
        return inputs, output_d

    def get_align_metrics(self) -> Dict[str, tf.keras.metrics.Metric]:
        d = self.probe_model_output["align_probe"]
        output_d = build_align_acc_dict_pairwise(d)
        return output_d

    def get_inference_model(self):
        return self.single_model

    def load_checkpoint(self, model_save_path):
        checkpoint = tf.train.Checkpoint(self.pair_model)
        checkpoint.restore(model_save_path).expect_partial()
