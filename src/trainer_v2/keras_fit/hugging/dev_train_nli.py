import sys

import tensorflow as tf
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.dataset_factories import get_classification_dataset
from trainer_v2.custom_loop.definitions import ModelConfigType
from trainer_v2.custom_loop.run_config2 import RunConfig2, get_run_config2
from trainer_v2.custom_loop.train_loop_helper import get_strategy_from_config
from trainer_v2.per_project.transparency.splade_regression.huggingface_helper import ignore_distilbert_save_warning
from trainer_v2.train_util.arg_flags import flags_parser
from transformers import TFAutoModelForSequenceClassification


class ModelConfig(ModelConfigType):
    max_seq_length = 300
    num_classes = 3
    model_type = "distilbert-base-uncased"


def get_classification_model(model_config: ModelConfig, is_training):
    input_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="input_ids")
    segment_ids = tf.keras.layers.Input(shape=(None,), dtype=tf.int32, name="segment_ids")
    attention_mask = tf.cast(tf.not_equal(input_ids, 0), tf.int32)
    new_inputs = {
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }
    model = TFAutoModelForSequenceClassification.from_pretrained(
        model_config.model_type, num_labels=model_config.num_classes)
    c_log.info("Initialize model parameter using huggingface: model_type=%s", model_config.model_type)
    output = model(new_inputs, training=is_training)
    new_model = tf.keras.models.Model(inputs=new_inputs, outputs=[output.logits])
    return new_model


def main(args):
    c_log.info(__file__)
    ignore_distilbert_save_warning()

    run_config: RunConfig2 = get_run_config2(args)
    run_config.print_info()
    model_config = ModelConfig()

    def build_dataset(input_files, is_for_training):
        return get_classification_dataset(
            input_files, run_config, model_config, is_for_training)

    strategy = get_strategy_from_config(run_config)
    with strategy.scope():
        optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        dataset = build_dataset(run_config.dataset_config.train_files_path, True)
        model = get_classification_model(model_config, True)
        model.compile(optimizer=optimizer, loss=loss)
        for i in range(5):
            model.fit(dataset, epochs=1, steps_per_epoch=4)
            model.save(run_config.train_config.model_save_path)



if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)


