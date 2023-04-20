
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from transformers import TFAutoModelForSequenceClassification

from trainer_v2.custom_loop.dataset_factories import get_classification_dataset, get_pairwise_dataset
from trainer_v2.custom_loop.definitions import ModelConfigType
from trainer_v2.custom_loop.train_loop_helper import get_strategy_from_config
import sys
from trainer_v2.chair_logging import c_log, IgnoreFilter, IgnoreFilterRE
from tensorflow.keras import layers, callbacks
import tensorflow as tf

from taskman_client.wrapper3 import report_run3
from trainer_v2.custom_loop.run_config2 import RunConfig2, get_run_config2
from trainer_v2.train_util.arg_flags import flags_parser


class ModelConfig(ModelConfigType):
    max_seq_length = 256
    num_classes = 1
    model_type = "bert-base-uncased"


def get_model(model_config: ModelConfig, run_config):
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
    optimizer = tf.keras.optimizers.Adam(learning_rate=run_config.train_config.learning_rate)
    new_model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.Hinge(),
        steps_per_execution=run_config.common_run_config.steps_per_execution,
    )

    return new_model


# Create a custom callback to print loss for each step
class LossPrintingCallback(callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        loss = logs.get("loss")
        if loss:
            c_log.info(f"Batch {batch}, Loss: {loss:.4f}")


@report_run3
def main(args):
    c_log.info(__file__)
    run_config: RunConfig2 = get_run_config2(args)
    run_config.print_info()
    model_config = ModelConfig()

    def build_dataset(input_files, is_for_training):
        return get_pairwise_dataset(
            input_files, run_config, model_config, is_for_training)

    run_name = str(run_config.common_run_config.run_name)
    c_log.info("Run name: %s", run_name)

    strategy = get_strategy_from_config(run_config)
    with strategy.scope():
        train_dataset = build_dataset(run_config.dataset_config.train_files_path, True)
        if run_config.dataset_config.eval_files_path:
            eval_dataset = build_dataset(run_config.dataset_config.eval_files_path, False)
        else:
            eval_dataset = None
        c_log.info("Building model")
        model = get_model(model_config, run_config)

        c_log.info("model.fit() train_step=%d", run_config.train_config.train_step)
        model.fit(train_dataset,
                  validation_data=eval_dataset,
                  epochs=1,
                  steps_per_epoch=run_config.train_config.train_step,
                  validation_steps=100,
                  )
        model.save(run_config.train_config.model_save_path)


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)


