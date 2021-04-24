import tensorflow as tf

from explain.explain_model import ExplainModeling
from explain.pairing.common_unit import linear, mlp, per_layer_modeling
from models.transformer.transformer_cls import transformer_pooled

Tensor = tf.Tensor


class LMS2Config:
    num_tags = 3
    use_embedding_out = True
    per_layer_component = 'linear'


class MatchPredictor3way:
    def __init__(self,
                 main_model: transformer_pooled,
                 ex_model: ExplainModeling,
                 per_layer_component: str,
                 num_e_signal: int,
                 use_embedding_out: bool
                 ):
        all_layers = main_model.model.get_all_encoder_layers()  # List[ tensor[batch,seq_length, hidden] ]
        if use_embedding_out:
            all_layers = [main_model.model.get_embedding_output()] + all_layers

        self.main_model = main_model
        _, input_mask, _ = main_model.x_list
        self.ex_model = ex_model
        if per_layer_component == 'linear':
            network = linear
        elif per_layer_component == 'mlp':
            network = mlp
        else:
            assert False

        self.per_layer_models_list = []
        for e_idx in range(num_e_signal):
            with tf.variable_scope("ex_predictor_{}".format(e_idx)):
                ex_scores = ex_model.get_ex_scores(e_idx)  # tensor[batch, seq_length]
                per_layer_models = list([per_layer_modeling(layer, ex_scores, input_mask, network)
                                         for layer_no, layer in enumerate(all_layers)])
                self.per_layer_models_list.append(per_layer_models)

        loss = 0
        all_losses = []
        per_layer_logits_list = []
        for per_layer_models in self.per_layer_models_list:
            for d in per_layer_models:
                loss += d.loss

            # Tensor[num_layer]
            all_losses_for_label = tf.stack([d.loss for d in per_layer_models])
            # List[Tensor[num_layer]]
            all_losses.append(all_losses_for_label)
            # List[Tensor[batch, 2]]
            per_layer_logits = [d.logits for d in per_layer_models]
            # List[List[Tensor[batch, 2]]]
            per_layer_logits_list.append(per_layer_logits)

        self.all_losses = tf.stack(all_losses)
        self.loss = loss
        self.per_layer_logits_list = per_layer_logits_list


def build_model(ex_modeling_class, hparam, lms_model_config: LMS2Config):
    main_model = transformer_pooled(hparam, hparam.vocab_size)
    ex_model = ex_modeling_class(main_model.model.sequence_output,
                                 hparam.seq_max,
                                 lms_model_config.num_tags,
                                 main_model.batch2feed_dict)

    match_predictor = MatchPredictor3way(main_model,
                                         ex_model,
                                         lms_model_config.per_layer_component,
                                         lms_model_config.num_tags,
                                         lms_model_config.use_embedding_out
                                         )
    return main_model, ex_model, match_predictor

