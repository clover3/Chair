import sys

import tensorflow as tf
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.dataset_factories import get_classification_dataset
from trainer_v2.custom_loop.definitions import ModelConfigType
from trainer_v2.custom_loop.per_task.classification_trainer import ClassificationTrainer
from trainer_v2.custom_loop.run_config2 import RunConfig2, get_run_config2
from trainer_v2.custom_loop.train_loop import tf_run
from trainer_v2.custom_loop.trainer_if import TrainerIF
from trainer_v2.custom_loop.prediction_trainer import ModelV2IF
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


class DistilBertClassification(ModelV2IF):
    def __init__(self, model_config):
        self.model_config = model_config
        self.model: tf.keras.models.Model = None

    def build_model(self, run_config):
        self.model = get_classification_model(self.model_config, run_config.is_training())

    def get_keras_model(self) -> tf.keras.models.Model:
        return self.model

    def init_checkpoint(self, model_path):
        if model_path is not None:
            c_log.warning("Model path {} is given, but not used")


def main(args):
    c_log.info(__file__)
    ignore_distilbert_save_warning()

    run_config: RunConfig2 = get_run_config2(args)
    run_config.print_info()
    model_config = ModelConfig()
    trainer: TrainerIF = ClassificationTrainer(
        run_config,
        DistilBertClassification(model_config)
    )

    def build_dataset(input_files, is_for_training):
        return get_classification_dataset(
            input_files, run_config, model_config, is_for_training)

    tf_run(run_config, trainer, build_dataset)


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)


