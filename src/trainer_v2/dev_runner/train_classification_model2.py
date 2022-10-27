import functools
import sys

import official.nlp
import official.nlp.bert.run_classifier
import tensorflow as tf
from official.modeling import performance
from official.nlp.bert import configs as bert_configs, bert_models

from cpath import get_bert_config_path
from trainer_v2.dev_runner.train_classification_model import build_dataset
from trainer_v2.keras_fit.config_helper import ModelConfig
from trainer_v2.keras_fit.modeling import get_optimizer
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

    strategy = get_strategy(args.use_tpu(), args.tpu_name)
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

    run_fn = official.nlp.bert.run_classifier.run_keras_compile_fit
    run_fn(args.output_dir,
           strategy,
           get_model_fn,
           train_input_fn,
           None,
           tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
           metric_fn,
           run_config.init_checkpoint,
           run_config.get_epochs(),
           run_config.steps_per_epoch,
           steps_per_loop=run_config.steps_per_execution,
           eval_steps=0,
           training_callbacks=get_custom_callback(run_config),
           )


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main2(args)
