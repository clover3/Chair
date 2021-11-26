import tensorflow as tf


def get_bert_input_dict(seq_length):
    encoder_inputs = dict(
        input_word_ids=tf.keras.layers.Input(shape=(seq_length,), dtype=tf.int32),
        input_mask=tf.keras.layers.Input(shape=(seq_length,), dtype=tf.int32),
        input_type_ids=tf.keras.layers.Input(shape=(seq_length,), dtype=tf.int32),
    )
    return encoder_inputs
