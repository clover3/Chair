import sys
from typing import Callable

import keras
import tensorflow as tf
from keras import Model

from cpath import get_bert_config_path
from models.keras_model.dev.from_hub import load_bert_model_by_hub
from tlm.model.base import BertConfig
from trainer_v2.arg_flags import flags_parser
from trainer_v2.chair_logging import c_log
from trainer_v2.get_tpu_strategy import get_strategy
from trainer_v2.partial_processing.config_helper import ModelConfig, get_run_config_nli_train
from trainer_v2.partial_processing.dev_modeling import get_optimizer
from trainer_v2.partial_processing.misc_helper import parse_input_files
from trainer_v2.run_config import RunConfigEx

keras = tf.keras


class BERT_CLS:
    def __init__(self, gs_folder_bert, config: ModelConfig):
        num_classes = config.num_classes
        model, pooled_output, sequence_output = load_bert_model_by_hub(gs_folder_bert, config.max_seq_length)
        self.seq_out = sequence_output
        output = tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax)(pooled_output)
        self.model: keras.Model = model
        self.pooler = pooled_output
        self.output = output


def run_train(define_model_fn: Callable, dataset, run_config):
    # Define model
    model: Model = define_model_fn()
    optimizer = get_optimizer(run_config)
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy('accuracy', dtype=tf.float32)]
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics,
                  steps_per_execution=run_config.steps_per_execution)
    model.summary()
    model.fit(dataset, epochs=run_config.get_epochs(), steps_per_epoch=run_config.steps_per_epoch)
    model.save_weights(run_config.model_save_path)

#
# def run_eval(model_config, checkpoint_path, input_files, max_seq_length, run_config):
#     bert_config = model_config.bert_config
#     inputs = define_bert_keras_inputs(max_seq_length)
#     bert_classifier: tf.keras.layers.Layer = BertClassifierLayer(bert_config, True, model_config.num_classes)
#     output = bert_classifier(inputs)
#     model: tf.keras.models.Model = tf.keras.models.Model(inputs=inputs, outputs=output)
#     trainer_v2.misc_common.load_weights(checkpoint_path)


def create_classifier_dataset(file_path, seq_length, batch_size, is_training):
    dataset = tf.data.TFRecordDataset(file_path)
    if is_training:
        dataset = dataset.shuffle(100)
        dataset = dataset.repeat()

    def decode_record(record):
        name_to_features = {
            'input_ids': tf.io.FixedLenFeature([seq_length], tf.int64),
            'input_mask': tf.io.FixedLenFeature([seq_length], tf.int64),
            'segment_ids': tf.io.FixedLenFeature([seq_length], tf.int64),
            'label_ids': tf.io.FixedLenFeature([], tf.int64),
        }
        return tf.io.parse_single_example(record, name_to_features)

    def _select_data_from_record(record):
        x = {
            'input_word_ids': record['input_ids'],
            'input_mask': record['input_mask'],
            'input_type_ids': record['segment_ids']
        }
        y = record['label_ids']
        return (x, y)

    dataset = dataset.map(decode_record,
                          num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(
        _select_data_from_record,
        num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=is_training)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


def build_dataset(model_config, input_files, run_config):
    dataset = create_classifier_dataset(
        tf.io.gfile.glob(input_files),
        model_config.max_seq_length,
        run_config.batch_size,
        is_training=run_config.is_training)

    return dataset


def get_define_model_fn(model_config, run_config) -> Callable:
    def define_model():
        bert_cls = BERT_CLS(run_config.init_checkpoint, model_config)
        model = bert_cls.model
        return model

    return define_model


def main(args):
    c_log.info("main {}".format(args))
    strategy = get_strategy(args.use_tpu, args.tpu_name)
    input_files = parse_input_files(args.input_files)
    bert_config = BertConfig.from_json_file(get_bert_config_path())
    # max_seq_length = 300
    run_config: RunConfigEx = get_run_config_nli_train(args)
    model_config = ModelConfig(bert_config, 128)
    define_model_fn = get_define_model_fn(model_config, run_config)
    with strategy.scope():
        dataset = build_dataset(model_config, input_files, run_config)
        # run_eval(model_config, checkpoint_path, input_files, max_seq_length, run_config)
        run_train(define_model_fn, dataset, run_config)


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)
