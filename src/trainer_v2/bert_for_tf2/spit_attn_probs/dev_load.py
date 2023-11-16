import tensorflow as tf


from cpath import get_bert_config_path
from trainer_v2.custom_loop.modeling_common.bert_common import load_bert_config
from trainer_v2.custom_loop.neural_network_def.combine2d import ReduceMaxLayer
from trainer_v2.custom_loop.neural_network_def.inferred_attention import InferredAttention
from trainer_v2.custom_loop.neural_network_def.segmented_enc import FuzzyLogicLayerNoSum
from trainer_v2.custom_loop.neural_network_def.siamese import ModelConfig2SegProject


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
    model.init_checkpoint()

if __name__ == "__main__":
    main()