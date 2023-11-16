
from cpath import get_bert_config_path
from trainer_v2.custom_loop.modeling_common.bert_common import load_bert_config
from trainer_v2.per_project.transparency.mmp.data_gen.tt_train_gen import get_convert_to_bow_qtw
from trainer_v2.per_project.transparency.mmp.tt_model.tt1 import TranslationTableInferenceQTW
from trainer_v2.per_project.transparency.mmp.tt_model.model_conf_defs import InputShapeConfigTT100_4
from trainer_v2.per_project.transparency.mmp.tt_model.encoders import DummyTermEncoder
import tensorflow as tf


def get_dummy_tt_scorer():
    convert_to_bow = get_convert_to_bow_qtw()
    input_shape = InputShapeConfigTT100_4()
    bert_params = load_bert_config(get_bert_config_path())
    term_encoder = DummyTermEncoder(bert_params)
    tti = TranslationTableInferenceQTW(bert_params, input_shape, term_encoder)
    def score_fn(query, document):
        inputs = []
        for text in [query, document]:
            tfs, input_ids, qtw = convert_to_bow(text)
            for l in [input_ids, tfs, qtw]:
                inputs.append(tf.expand_dims(tf.constant(l), 0))
        output = tti.model(inputs)
        return output.numpy()[0]

    return score_fn


def main():
    input_shape = InputShapeConfigTT100_4()
    bert_params = load_bert_config(get_bert_config_path())

    model_path = "C:\work\code\chair\output\model\\runs\\tt1\\model_1"
    model = tf.keras.models.load_model(model_path)

    convert_to_bow = get_convert_to_bow_qtw()
    term_encoder = model.layers[6]
    tti = TranslationTableInferenceQTW(bert_params, input_shape, term_encoder)
    tfs, input_ids, qtw = convert_to_bow("Where is bookstore")
    inputs = []
    for l in [input_ids, tfs, qtw, input_ids, tfs, qtw]:
        inputs.append(tf.expand_dims(tf.constant(l), 0))
    output = tti.model(inputs)
    print(output)


if __name__ == "__main__":
    main()