import bert
import tensorflow as tf
from tensorflow import keras

from cpath import get_bert_config_path
from trainer_v2.custom_loop.modeling_common.bert_common import load_bert_config


def main():
    bert_params = load_bert_config(get_bert_config_path())
    l_bert = bert.BertModelLayer.from_params(bert_params)
    l_bert2 = bert.BertModelLayer.from_params(bert_params)

    max_seq_len = 128
    l_input_ids = keras.layers.Input(shape=(max_seq_len,), dtype='int32', name="input_ids")
    l_token_type_ids = keras.layers.Input(shape=(max_seq_len,), dtype='int32', name="segment_ids")

    rep1 = l_bert([l_input_ids, l_token_type_ids])
    rep2 = l_bert2([l_input_ids, l_token_type_ids])

    concat_layer = tf.keras.layers.Concatenate()
    feature_rep = concat_layer([rep1, rep2])

    hidden = tf.keras.layers.Dense(bert_params.hidden_size, activation='relu')(feature_rep)
    output = tf.keras.layers.Dense(3, activation=tf.nn.softmax)(hidden)
    inputs = (l_input_ids, l_token_type_ids)
    model = keras.Model(inputs=inputs, outputs=output, name="bert_model")

    print(l_bert)
    for v in l_bert.variables:
        print(v.name)
    print(l_bert2)
    for v in l_bert2.variables:
        print(v.name)


if __name__ == "__main__":
    main()