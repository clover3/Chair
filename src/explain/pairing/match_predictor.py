from abc import ABC

import tensorflow as tf

from explain.explain_model import ExplainModeling
from explain.pairing.common_unit import linear, mlp, per_layer_modeling
from models.transformer.transformer_cls import transformer_pooled


class MatchPredictor:
    def __init__(self,
                 main_model: transformer_pooled,
                 ex_model: ExplainModeling,
                 target_ex_idx: int,
                 per_layer_component: str,
                 use_embedding_out: bool
                 ):
        all_layers = main_model.model.get_all_encoder_layers()  # List[ tensor[batch,seq_length, hidden] ]
        if use_embedding_out:
            all_layers = [main_model.model.get_embedding_output()] + all_layers

        self.main_model = main_model
        _, input_mask, _ = main_model.x_list
        self.ex_model = ex_model
        ex_scores = ex_model.get_ex_scores(target_ex_idx)  # tensor[batch, seq_length]
        if per_layer_component == 'linear':
            network = linear
        elif per_layer_component == 'mlp':
            network = mlp
        else:
            assert False

        with tf.variable_scope("match_predictor"):
            per_layer_models = list([per_layer_modeling(layer, ex_scores, input_mask, network)
                                     for layer_no, layer in enumerate(all_layers)])
        self.per_layer_models = per_layer_models

        loss = 0
        for d in per_layer_models:
            loss += d.loss

        self.all_losses = tf.stack([d.loss for d in per_layer_models])
        self.loss = loss
        self.per_layer_logits = list([d.logits for d in per_layer_models])


class LMSConfigI(ABC):
    num_tags = None
    target_idx = None
    use_embedding_out = None
    per_layer_component = None


def build_model(ex_modeling_class, hparam, lms_model_config: LMSConfigI):
    main_model = transformer_pooled(hparam, hparam.vocab_size)
    ex_model = ex_modeling_class(main_model.model.sequence_output,
                                 hparam.seq_max,
                                 lms_model_config.num_tags,
                                 main_model.batch2feed_dict)

    match_predictor = MatchPredictor(main_model,
                                     ex_model,
                                     lms_model_config.target_idx,
                                     lms_model_config.per_layer_component,
                                     lms_model_config.use_embedding_out
                                     )
    return main_model, ex_model, match_predictor


class LMSConfig(LMSConfigI):
    num_tags = 3
    target_idx = 2
    use_embedding_out = False
    per_layer_component = 'linear'


class LMSConfig2(LMSConfigI):
    num_tags = 3
    target_idx = 1
    use_embedding_out = True
    per_layer_component = 'linear'
