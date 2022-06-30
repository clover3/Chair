import tensorflow as tf
from tensorflow import keras

from abrl.log_regression_models import get_shape_list2
from cpath import get_bert_config_path
from trainer_v2.bert_for_tf2 import BertModelLayer
from trainer_v2.bert_for_tf2.w_mask.model_bert_mask import load_stock_weights_nc
from trainer_v2.custom_loop.modeling_common.bert_common import load_bert_config, load_bert_checkpoint, BERT_CLS
from trainer_v2.custom_loop.modeling_common.network_utils import VectorThreeFeature, TwoLayerDense
from trainer_v2.custom_loop.neural_network_def.combine2d import get_new_input_ids, \
    expand_tile, ReduceMaxLayer
from trainer_v2.custom_loop.neural_network_def.multi_segments import create_attention_mask
from trainer_v2.custom_loop.neural_network_def.segmented_enc import FuzzyLogicLayerNoSum
from trainer_v2.custom_loop.neural_network_def.siamese import ModelConfig2SegProject


class InferredAttention:
    def __init__(self, combine_ld1_layer, combine_ld2_layer):
        super(InferredAttention, self).__init__()
        self.combine_ld1_layer = combine_ld1_layer
        self.combine_ld2_layer = combine_ld2_layer

    def build_model(self, bert_params,
                    ref_encoder1, ref_encoder2,
                    config: ModelConfig2SegProject):
        Dense = tf.keras.layers.Dense
        bert_params.use_position_embeddings = False
        # ref_encoder1 = BertModelLayerSAP.from_params(bert_params, name="encoder1/bert")
        # ref_encoder2 = BertModelLayerSAP.from_params(bert_params, name="encoder2/bert")
        ref_encoder1.trainable = False
        ref_encoder2.trainable = False

        max_seq_len1 = config.max_seq_length1
        max_seq_len2 = config.max_seq_length2
        num_classes = config.num_classes

        l_input_ids1 = keras.layers.Input(shape=(max_seq_len1,), dtype='int32', name="input_ids1")
        l_token_type_ids1 = keras.layers.Input(shape=(max_seq_len1,), dtype='int32', name="segment_ids1")
        input_mask1 = tf.cast(tf.not_equal(l_input_ids1, 0), tf.int32)

        l_input_ids2 = keras.layers.Input(shape=(max_seq_len2,), dtype='int32', name="input_ids2")
        l_token_type_ids2 = keras.layers.Input(shape=(max_seq_len2,), dtype='int32', name="segment_ids2")
        input_mask2 = tf.cast(tf.not_equal(l_input_ids2, 0), tf.int32)

        B, _ = get_shape_list2(l_input_ids1)

        attention_mask1 = create_attention_mask(l_input_ids1, input_mask1)
        attention_mask2 = create_attention_mask(l_input_ids2, input_mask2)
        _, atten_probs1 = ref_encoder1([l_input_ids1, l_token_type_ids1], attention_mask1)
        _, atten_probs2 = ref_encoder2([l_input_ids2, l_token_type_ids2], attention_mask2)

        def simplify_attn_probs(attn_probs):
            t = tf.reduce_mean(tf.stack(attn_probs, axis=1), axis=1)
            return tf.reduce_mean(t, axis=1)


        attention_probs1 = simplify_attn_probs(atten_probs1)
        attention_probs2 = simplify_attn_probs(atten_probs2)  # [B, L, L]

        m = 3
        new_input_ids1 = get_new_input_ids(l_input_ids1, attention_probs1, max_seq_len1, m)
        new_input_ids2 = get_new_input_ids(l_input_ids2, attention_probs2, max_seq_len2, m)

        flat_input_ids1 = tf.reshape(new_input_ids1, [B * max_seq_len1, 1 + m])
        flat_input_ids2 = tf.reshape(new_input_ids2, [B * max_seq_len2, 1 + m])

        def build_bert_cls(prefix):
            l_bert = BertModelLayer.from_params(bert_params, name="{}/bert".format(prefix))
            pooler = tf.keras.layers.Dense(bert_params.hidden_size, activation=tf.nn.tanh, name="{}/bert/pooler/dense".format(prefix))
            return BERT_CLS(l_bert, pooler)

        bert_cls1 = build_bert_cls("encoder1_n")
        bert_cls2 = build_bert_cls("encoder2_n")

        rep1 = bert_cls1.apply(flat_input_ids1)  # [B * L, H]
        rep2 = bert_cls2.apply(flat_input_ids2)

        rep1_3d = tf.reshape(rep1, [B, max_seq_len1, bert_params.hidden_size])
        rep2_3d = tf.reshape(rep2, [B, max_seq_len2, bert_params.hidden_size])

        rep1_4d = expand_tile(rep1_3d, 2, 3, max_seq_len2)  # [B, L1, L2, H]
        rep2_4d = expand_tile(rep2_3d, 1, 3, max_seq_len1)  # [B, L1, L2, H]
        vtf = VectorThreeFeature()
        feature_rep = vtf((rep1_4d, rep2_4d))

        local_decisions = TwoLayerDense(bert_params.hidden_size, num_classes,
                                        name="local_decision")(feature_rep) # [B, L1, L2, C]

        ld = self.combine_ld1_layer()(local_decisions)
        self.local_decisions = local_decisions
        output = self.combine_ld2_layer()([ld, input_mask2])
        inputs = (l_input_ids1, l_token_type_ids1, l_input_ids2, l_token_type_ids2)
        model = keras.Model(inputs=inputs, outputs=output, name="bert_model")
        self.model: keras.Model = model
        self.l_bert_list = [bert_cls1, bert_cls2]
        self.ref_encoders = [ref_encoder1, ref_encoder2]

    def get_keras_model(self):
        return self.model

    def init_checkpoint(self, init_checkpoint_s):
        bert_checkpoint, model_checkpoint = init_checkpoint_s.split(",")
        l_bert1, l_bert2 = self.l_bert_list
        load_bert_checkpoint(l_bert1, bert_checkpoint)
        load_bert_checkpoint(l_bert2, bert_checkpoint)

        load_stock_weights_nc(l_bert1, model_checkpoint, n_expected_restore=76)
        load_stock_weights_nc(l_bert2, model_checkpoint, n_expected_restore=76)

        for e in self.ref_encoders:
            print(e)


def main():
    model_checkpoint = "model/nli_ts_run84_1/model_10"
    bert_params = load_bert_config(get_bert_config_path())
    ref_model = tf.keras.models.load_model(model_checkpoint)

    for layer in ref_model.layers:
        if layer.name == "encoder1/bert":
            ref_encoder1 = layer
        elif layer.name == "encoder2/bert":
            ref_encoder2 = layer
    model_config = ModelConfig2SegProject()
    model = InferredAttention(ReduceMaxLayer, FuzzyLogicLayerNoSum)
    model.build_model(bert_params, ref_encoder1, ref_encoder2, model_config)


if __name__ == "__main__":
    main()