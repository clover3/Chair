import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification

from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.definitions import ModelConfigType


class ModelConfig(ModelConfigType):
    max_seq_length = 256
    num_classes = 1
    model_type = "bert-base-uncased"


def get_transformer_pairwise_model(model_config: ModelConfig, run_config, optimizer_factory=None):
    is_training = run_config.is_training()
    input_ids1 = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="input_ids1")
    segment_ids1 = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="token_type_ids1")
    input_ids2 = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="input_ids2")
    segment_ids2 = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="token_type_ids2")
    input_1 = {
        'input_ids': input_ids1,
        'token_type_ids': segment_ids1
    }
    input_2 = {
        'input_ids': input_ids2,
        'token_type_ids': segment_ids2
    }
    model = TFAutoModelForSequenceClassification.from_pretrained(
        model_config.model_type, num_labels=1)
    c_log.info("Initialize model parameter using huggingface: model_type=%s", model_config.model_type)

    def network(bert_input):
        t = model.bert(bert_input, training=is_training)
        pooled_output = t[1]
        t = model.dropout(pooled_output, training=is_training)
        return model.classifier(t)

    logits1 = network(input_1)
    logits2 = network(input_2)

    inputs = [input_ids1, segment_ids1, input_ids2, segment_ids2 ,]
    pred = logits1 - logits2
    new_model = tf.keras.models.Model(inputs=inputs, outputs=pred)
    if optimizer_factory is not None:
        optimizer = optimizer_factory(learning_rate=run_config.train_config.learning_rate)
        new_model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.Hinge(),
            steps_per_execution=run_config.common_run_config.steps_per_execution,
        )

    return new_model

