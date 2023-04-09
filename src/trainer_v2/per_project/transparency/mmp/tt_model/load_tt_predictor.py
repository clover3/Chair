from typing import List

from cpath import get_bert_config_path
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.modeling_common.bert_common import load_bert_config
from trainer_v2.per_project.transparency.mmp.data_gen.tt_train_gen import get_convert_to_bow_qtw
from trainer_v2.per_project.transparency.mmp.tt_model.tt1 import InputShapeConfigTT100_4, TranslationTableInferenceQTW, \
    DummyTermEncoder
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

def get_tt_scorer(model_path):
    SpecI = tf.TensorSpec([None], dtype=tf.int32)
    SpecF = tf.TensorSpec([None], dtype=tf.float32)
    convert_to_bow = get_convert_to_bow_qtw()
    input_shape = InputShapeConfigTT100_4()
    bert_params = load_bert_config(get_bert_config_path())
    c_log.info("Loading model from %s", model_path)
    tt_model = tf.keras.models.load_model(model_path)
    term_encoder = get_term_encoder_from_model_by_shape(
        tt_model, bert_params.hidden_size)
    sig = (SpecI, SpecI, SpecF, SpecI, SpecI, SpecF),

    def score_fn(qd_list: List):
        def generator():
            for query, document in qd_list:
                x = []
                for text in [query, document]:
                    tfs, input_ids, qtw = convert_to_bow(text)
                    for l in [input_ids, tfs, qtw]:
                        x.append(tf.constant(l))
                yield tuple(x),

        dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=sig)
        dataset = dataset.batch(16)
        output = tti.model.predict(dataset)
        return output

    c_log.info("Defining network")
    tti = TranslationTableInferenceQTW(bert_params, input_shape, term_encoder)
    return score_fn

