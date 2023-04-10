import tensorflow as tf
from official import nlp
from official.modeling import tf_utils

from trainer_v2.keras_fit.bert_encoder import MyBertEncoder
from trainer_v2.run_config import RunConfigEx


def get_transformer_encoder(bert_config,
                            input_prefix_fix="",
                            ):
    def build_keras_input(name):
        return tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name=input_prefix_fix + name)
    mask = build_keras_input('input_mask')
    word_ids = build_keras_input('input_word_ids')
    type_ids = build_keras_input('input_type_ids')

    return bert_encoder_factory(bert_config, word_ids, mask, type_ids)


def bert_encoder_factory(bert_config, word_ids, mask, type_ids):
    kwargs = dict(
        vocab_size=bert_config.vocab_size,
        hidden_size=bert_config.hidden_size,
        num_layers=bert_config.num_hidden_layers,
        num_attention_heads=bert_config.num_attention_heads,
        inner_dim=bert_config.intermediate_size,
        inner_activation=tf_utils.get_activation(bert_config.hidden_act),
        output_dropout=bert_config.hidden_dropout_prob,
        attention_dropout=bert_config.attention_probs_dropout_prob,
        max_sequence_length=bert_config.max_position_embeddings,
        type_vocab_size=bert_config.type_vocab_size,
        embedding_width=bert_config.embedding_size,
        initializer=tf.keras.initializers.TruncatedNormal(
            stddev=bert_config.initializer_range))
    return MyBertEncoder(word_ids,
                         mask,
                         type_ids, **kwargs)


def get_optimizer(run_config: RunConfigEx):
    num_train_steps = run_config.train_step
    warmup_steps = int(num_train_steps * 0.1)
    optimizer = nlp.optimization.create_optimizer(
        run_config.learning_rate, num_train_steps=num_train_steps, num_warmup_steps=warmup_steps)
    return optimizer


