
import tensorflow as tf
from tensorflow import keras

from models.transformer.bert_common_v2 import get_shape_list2
from trainer_v2.bert_for_tf2 import BertModelLayer
from trainer_v2.custom_loop.definitions import ModelConfig300_3
from trainer_v2.custom_loop.modeling_common.bert_common import BERT_CLS, load_bert_checkpoint
from trainer_v2.custom_loop.modeling_common.network_utils import TwoLayerDense
from trainer_v2.custom_loop.neural_network_def.inner_network import BertBasedModelIF


class SpanLevelReasoning(BertBasedModelIF):
    def __init__(self):
        super(SpanLevelReasoning, self).__init__()

    def build_model(self, bert_params, config: ModelConfig300_3):
        prefix = "encoder"
        l_bert = BertModelLayer.from_params(bert_params, name="{}/bert".format(prefix))
        pooler = tf.keras.layers.Dense(bert_params.hidden_size, activation=tf.nn.tanh,
                                       name="{}/bert/pooler/dense".format(prefix))
        bert_cls = BERT_CLS(l_bert, pooler)

        max_seq_len = config.max_seq_length
        num_classes = config.num_classes

        l_input_ids1 = keras.layers.Input(shape=(max_seq_len1,), dtype='int32', name="input_ids1")
        l_token_type_ids1 = keras.layers.Input(shape=(max_seq_len1,), dtype='int32', name="segment_ids1")
        stacked_input_ids = NotImplemented
        stacked_segment_ids = NotImplemented

        feature_rep = bert_cls.apply([stacked_input_ids, stacked_segment_ids])
        # [B, M, H], M is number of span / H is hidden size
        Dense = tf.keras.layers.Dense

        # Entail / Neutral / Contradiction
        y_n = [0, 1, 0]
        y_c = [0, 0, 1]

        mid_labels = ["N", "C"]
        probs_out = []
        local_decisions = []
        for mid_label in mid_labels:
            attn_raw = TwoLayerDense(bert_params.hidden_size, 1,
                                     activation1=tf.nn.tanh,
                                     activation2=tf.nn.sigmoid,
                                     name=f"attn_{mid_label}")(feature_rep)  # [B, M, 1]
            logit_raw = Dense(1)(feature_rep)
            norm = tf.reduce_sum(attn_raw, axis=1, keepdims=True)
            attn_normed = tf.divide(attn_raw, norm)
            logit_out = logit_raw * attn_normed  # L_n
            probs = Dense(1, activation=tf.nn.sigmoid)(logit_out)
            local_decisions.append(logit_raw)
            probs_out.append(probs)

        B, _ = get_shape_list2(l_input_ids1)
        self.local_decisions = local_decisions
        inputs = (l_input_ids1, l_token_type_ids1, l_input_ids2, l_token_type_ids2)
        model = keras.Model(inputs=inputs, outputs=probs_out, name="bert_model")
        self.model: keras.Model = model
        self.bert_cls = bert_cls

    def get_keras_model(self):
        return self.model

    def init_checkpoint(self, init_checkpoint):
        load_bert_checkpoint(self.bert_cls, init_checkpoint)

