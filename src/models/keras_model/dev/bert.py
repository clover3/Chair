import os

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

tfds.disable_progress_bar()

from official.nlp import bert

# Load the required submodules
import official.nlp.bert.bert_models
import official.nlp.bert.configs
import json


def encode_sentence(s, tokenizer):
    tokens = list(tokenizer.tokenize(s))
    tokens.append('[SEP]')
    return tokenizer.convert_tokens_to_ids(tokens)


def tokenize_convert(glue_dict, tokenizer):
    num_examples = len(glue_dict["sentence1"])

    sentence1 = tf.ragged.constant([
        encode_sentence(s, tokenizer)
        for s in np.array(glue_dict["sentence1"])])
    sentence2 = tf.ragged.constant([
        encode_sentence(s, tokenizer)
        for s in np.array(glue_dict["sentence2"])])

    cls = [tokenizer.convert_tokens_to_ids(['[CLS]'])]*sentence1.shape[0]
    input_word_ids = tf.concat([cls, sentence1, sentence2], axis=-1)

    input_mask = tf.ones_like(input_word_ids).to_tensor()

    type_cls = tf.zeros_like(cls)
    type_s1 = tf.zeros_like(sentence1)
    type_s2 = tf.ones_like(sentence2)
    input_type_ids = tf.concat(
        [type_cls, type_s1, type_s2], axis=-1).to_tensor()

    inputs = {
        'input_word_ids': input_word_ids.to_tensor(),
        'input_mask': input_mask,
        'input_type_ids': input_type_ids}

    return inputs

def maybe_load_bert():
    hub_url_bert = "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3"
    bert_something = tf.keras.models.load_model(hub_url_bert)
    # config_dict = json.loads(tf.io.gfile.GFile(bert_config_file).read())
    # bert_config = bert.configs.BertConfig.from_dict(config_dict)
    # bert_classifier, bert_encoder = bert.bert_models.classifier_model(
    #     bert_config, num_labels=2)
    print(bert_something)
    return bert_something
    # checkpoint = tf.train.Checkpoint(encoder=bert_encoder)
    # tf.keras.utils.plot_model(bert_encoder, show_shapes=True, dpi=48)

    # checkpoint.read(
    #     os.path.join(gs_folder_bert, "variables", 'variables')).assert_consumed()


def main():
    glue, info = tfds.load('glue/mrpc', with_info=True,
                           # It's small, load the whole dataset
                           batch_size=-1)
    print(list(glue.keys()))
    print(info.features)
    print(info.features['label'].names)
    maybe_load_bert()
    print("Vocab size:", len(tokenizer.vocab))

    glue_train = tokenize_convert(glue['train'], tokenizer)
    glue_train_labels = glue['train']['label']

    glue_validation = tokenize_convert(glue['validation'], tokenizer)
    glue_validation_labels = glue['validation']['label']

    glue_test = tokenize_convert(glue['test'], tokenizer)
    glue_test_labels = glue['test']['label']

    bert_config_file = os.path.join(gs_folder_bert, "bert_config.json")
    config_dict = json.loads(tf.io.gfile.GFile(bert_config_file).read())

    bert_config = bert.configs.BertConfig.from_dict(config_dict)
    bert_classifier, bert_encoder = bert.bert_models.classifier_model(
        bert_config, num_labels=2)
    glue_batch = {key: val[:10] for key, val in glue_train.items()}
    some_output = bert_classifier(
        glue_batch, training=True
    ).numpy()
    checkpoint = tf.train.Checkpoint(encoder=bert_encoder)
    # tf.keras.utils.plot_model(bert_encoder, show_shapes=True, dpi=48)
    print("checkpoint.read")
    checkpoint.read(
        os.path.join(gs_folder_bert, "variables", 'variables')).assert_consumed()


if __name__ == "__main__":
    main()