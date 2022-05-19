import numpy as np
from official.nlp.bert.bert_models import classifier_model

from trainer_v2.partial_processing.config_helper import get_bert_config


def main():
    bert_config = get_bert_config()
    # bert_encoder: tf.keras.Model = get_transformer_encoder(
    #     bert_config,
    # )
    max_seq_length = 512
    model, bert_encoder = classifier_model(bert_config, 3, max_seq_length)
    strategy = get_strategy(False, "")
    with strategy.scope():
        optimizer = get_optimizer(RunConfigEx(train_step=10))
        bert_encoder.optimizer = performance.configure_optimizer(
            optimizer)

        bert_v2 = "C:\\work\\Code\\Chair\\output\\model\\runs\\keras_bert\\uncased_L-12_H-768_A-12\\bert_model.ckpt"
        checkpoint = tf.train.Checkpoint(model=bert_encoder)
        checkpoint.restore(bert_v2).assert_existing_objects_matched()

        words_ids = [101, 1009, 1010, 101]
        segment_ids = [0] * 4
        mask = [0] * 4
        pad_len = max_seq_length - 4

        input_items = [np.array(item) for item in [words_ids, segment_ids, mask]]
        input_items = [np.pad(item, (0, pad_len)) for item in input_items]
        input_items = [np.expand_dims(item, 0) for item in input_items]
        item = bert_encoder(tuple(input_items))
        print(len(item))

import functools
import sys

import official.nlp
import official.nlp.bert.run_classifier
import tensorflow as tf
from official.modeling import performance
from official.nlp.bert import configs as bert_configs, bert_models

from cpath import get_bert_config_path
from trainer_v2.dev_runner.train_classification_model import build_dataset
from trainer_v2.partial_processing.config_helper import ModelConfig
from trainer_v2.partial_processing.modeling import get_optimizer
from trainer_v2.run_config import RunConfigEx, get_run_config_nli_train
from trainer_v2.train_util.arg_flags import flags_parser
from trainer_v2.train_util.callbacks import get_custom_callback
from trainer_v2.train_util.get_tpu_strategy import get_strategy
from trainer_v2.train_util.misc_helper import parse_input_files

def main2(args):
    input_files = parse_input_files(args.input_files)
    bert_config = bert_configs.BertConfig.from_json_file(get_bert_config_path())
    max_seq_length = 300
    run_config: RunConfigEx = get_run_config_nli_train(args)
    model_config = ModelConfig(bert_config, max_seq_length)

    strategy = get_strategy(args.use_tpu, args.tpu_name)
    def get_model_fn():
        classifier_model, core_model = \
            bert_models.classifier_model(bert_config, model_config.num_classes, max_seq_length)
        optimizer = get_optimizer(run_config)
        classifier_model.optimizer = performance.configure_optimizer(
            optimizer)
        return classifier_model, core_model

    def train_input_fn():
        dataset = build_dataset(model_config, input_files, run_config)
        return dataset

    metric_fn = functools.partial(
        tf.keras.metrics.SparseCategoricalAccuracy,
        'accuracy',
        dtype=tf.float32)
    # with strategy.scope():
    #     bert_model, sub_model = get_model_fn()
    #     optimizer = bert_model.optimizer
    #
    #     checkpoint = tf.train.Checkpoint(model=sub_model)
        # checkpoint.restore(run_config.init_checkpoint).assert_existing_objects_matched()
    #
    run_fn = official.nlp.bert.run_classifier.run_keras_compile_fit
    run_fn(args.output_dir,
           strategy,
           get_model_fn,
           train_input_fn,
           None,
           None,
           None,
           run_config.init_checkpoint,
           0,
           run_config.steps_per_epoch,
           steps_per_loop=run_config.steps_per_execution,
           eval_steps=0,
           training_callbacks=get_custom_callback(run_config),
           )


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main2(args)
