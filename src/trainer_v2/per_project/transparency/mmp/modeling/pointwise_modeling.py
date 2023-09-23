import tensorflow as tf
from transformers import TFAutoModelForSequenceClassification
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.definitions import HFModelConfigType


def get_transformer_pointwise_model(model_config: HFModelConfigType, run_config, optimizer_factory=None):
    is_training = run_config.is_training()
    input_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="input_ids")
    segment_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="token_type_ids")
    label_ids = tf.keras.layers.Input(shape=(1,), dtype=tf.int32, name="label_ids")
    inputs = {
        'input_ids': input_ids,
        'token_type_ids': segment_ids
    }
    model = TFAutoModelForSequenceClassification.from_pretrained(
        model_config.model_type, num_labels=1)
    c_log.info("Initialize model parameter using huggingface: model_type=%s", model_config.model_type)

    def network(bert_input):
        t = model.bert(bert_input, training=is_training)
        pooled_output = t[1]
        t = model.dropout(pooled_output, training=is_training)
        return model.classifier(t)

    logits = network(inputs)
    inputs = [input_ids, segment_ids, label_ids]
    pred = logits
    new_model = tf.keras.models.Model(inputs=inputs, outputs=pred)
    if optimizer_factory is not None:
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        optimizer = optimizer_factory(learning_rate=run_config.train_config.learning_rate)
        new_model.compile(
            optimizer=optimizer,
            loss=loss,
            steps_per_execution=run_config.common_run_config.steps_per_execution,
        )
    return new_model
