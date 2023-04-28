from cpath import get_bert_config_path
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.modeling_common.bert_common import load_bert_config
from trainer_v2.per_project.transparency.mmp.tt_model.net_common import get_tt_scorer, find_layer
from trainer_v2.per_project.transparency.mmp.tt_model.tt1 import TranslationTableInferenceQTW, \
    TTInfQTWAsym, ScoringLayer2, ScoringLayerSigmoidCap
from trainer_v2.per_project.transparency.mmp.tt_model.model_conf_defs import InputShapeConfigTT100_4
import tensorflow as tf


def get_term_encoder_from_model_by_shape(model, hidden_size):
    for idx, layer in enumerate(model.layers):
        try:
            shape = layer.output.shape
            if shape[1] == 100 and shape[2] == hidden_size:
                c_log.debug("Maybe this is local decision layer: {}".format(layer.name))
                return layer
        except AttributeError:
            print("layer is actually : ", layer)
        except IndexError:
            pass

    c_log.error("Layer not found")
    for idx, layer in enumerate(model.layers):
        c_log.error(idx, layer, layer.output.shape)
    raise KeyError


def SpecI():
    return tf.TensorSpec([None], dtype=tf.int32)


def load_translation_table_inference_qtw(model_path):
    input_shape = InputShapeConfigTT100_4()
    bert_params = load_bert_config(get_bert_config_path())
    c_log.info("Loading model from %s", model_path)
    tt_model = tf.keras.models.load_model(model_path)
    term_encoder = get_term_encoder_from_model_by_shape(
        tt_model, bert_params.hidden_size)
    tti = TranslationTableInferenceQTW(bert_params, input_shape, term_encoder)
    return tti


def get_tt_scorer_1(model_path):
    input_shape = InputShapeConfigTT100_4()
    bert_params = load_bert_config(get_bert_config_path())
    c_log.info("Loading model from %s", model_path)
    tt_model = tf.keras.models.load_model(model_path)
    term_encoder = get_term_encoder_from_model_by_shape(
        tt_model, bert_params.hidden_size)
    tti = TranslationTableInferenceQTW(bert_params, input_shape, term_encoder)
    score_fn = get_tt_scorer(tti.model)

    c_log.info("Defining network")
    return score_fn


def get_tt_scorer_6(model_path):
    input_shape = InputShapeConfigTT100_4()
    bert_params = load_bert_config(get_bert_config_path())
    c_log.info("Loading model from %s", model_path)
    paired_model = tf.keras.models.load_model(model_path)
    q_encoder = find_layer(paired_model, "term_encoder")
    d_encoder = find_layer(paired_model, "term_encoder_1")
    def scoring_layer_factory():
        return ScoringLayer2(bert_params.hidden_size)
    tti = TTInfQTWAsym(bert_params, input_shape, q_encoder, d_encoder, scoring_layer_factory)
    score_fn = get_tt_scorer(tti.model)

    c_log.info("Defining network")
    return score_fn


def get_tt_vector_scorer(model_path):
    input_shape = InputShapeConfigTT100_4()
    bert_params = load_bert_config(get_bert_config_path())
    c_log.info("Loading model from %s", model_path)
    paired_model = tf.keras.models.load_model(model_path)
    q_encoder = find_layer(paired_model, "term_vector")
    d_encoder = find_layer(paired_model, "term_vector_1")
    def scoring_layer_factory():
        return ScoringLayer2(bert_params.hidden_size)
    tt_v_inf = TTInfQTWAsym(bert_params, input_shape, q_encoder, d_encoder, scoring_layer_factory)
    score_fn = get_tt_scorer(tt_v_inf.model)
    c_log.info("Defining network")
    return score_fn


def get_tt9_scorer(model_path):
    input_shape = InputShapeConfigTT100_4()
    bert_params = load_bert_config(get_bert_config_path())
    c_log.info("Loading model from %s", model_path)
    paired_model = tf.keras.models.load_model(model_path)
    q_encoder = find_layer(paired_model, "term_vector")
    d_encoder = find_layer(paired_model, "term_vector_1")
    def scoring_layer_factory():
        return ScoringLayerSigmoidCap(bert_params.hidden_size)
    tt_v_inf = TTInfQTWAsym(bert_params, input_shape, q_encoder, d_encoder, scoring_layer_factory)
    score_fn = get_tt_scorer(tt_v_inf.model)
    c_log.info("Defining network")
    return score_fn

