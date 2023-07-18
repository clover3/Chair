import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub

import trainer_v2.per_project.transparency.mmp.alignment.galign_inf_helper
from models.keras_model.dev.bert import encode_sentence

tfds.disable_progress_bar()

from official.nlp import bert

# Load the required submodules


def tokenize_convert(sentence_list, tokenizer):
    sentence1 = tf.ragged.constant([
        encode_sentence(s, tokenizer)
        for s in np.array(sentence_list)])

    cls = [tokenizer.convert_tokens_to_ids(['[CLS]'])]*sentence1.shape[0]
    input_word_ids = tf.concat([cls, sentence1], axis=-1)

    input_mask = tf.ones_like(input_word_ids).to_tensor()

    type_cls = tf.zeros_like(cls)
    type_s1 = tf.zeros_like(sentence1)
    input_type_ids = tf.concat(
        [type_cls, type_s1], axis=-1).to_tensor()

    inputs = {
        'input_word_ids': input_word_ids.to_tensor(),
        'input_mask': input_mask,
        'input_type_ids': input_type_ids}

    return inputs


def get_mrpc_dataset(seq_length, batch_size=32):
    train_data_output_path = "C:\\work\\Code\\Chair\\src\\models\\keras_model\\dev\\mrpc_train.tf_record"
    training_dataset = trainer_v2.per_project.transparency.mmp.alignment.galign_inf_helper.build_dataset(
        train_data_output_path,
        seq_length,
        batch_size,
        is_training=False)()
    return training_dataset


def load_bert_model_by_hub(gs_folder_bert, seq_length):
    encoder = hub.KerasLayer(gs_folder_bert, trainable=True)
    print("encoder", encoder)
    print(f"The Hub encoder has {len(encoder.trainable_variables)} trainable variables")
    encoder_inputs = dict(
        input_word_ids=tf.keras.layers.Input(shape=(seq_length,), dtype=tf.int32),
        input_mask=tf.keras.layers.Input(shape=(seq_length,), dtype=tf.int32),
        input_type_ids=tf.keras.layers.Input(shape=(seq_length,), dtype=tf.int32),
    )
    print("encoder_inputs")

    outputs = encoder(encoder_inputs)
    pooled_output = outputs["pooled_output"]  # [batch_size, 768].
    sequence_output = outputs["sequence_output"]  # [batch_size, seq_length, 768].
    model = tf.keras.Model(encoder_inputs, pooled_output)
    return model, pooled_output, sequence_output


def main():
    return NotImplemented


if __name__ == "__main__":
    main()