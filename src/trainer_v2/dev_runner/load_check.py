import sys

import tensorflow as tf

from cpath import get_bert_config_path
from models.keras_model.bert_keras.modular_bert import BertClassifierLayer
from tlm.model.base import BertConfig
from trainer_v2.arg_flags import flags_parser
from trainer_v2.chair_logging import c_log
from trainer_v2.checkpoint_util import load_weights
from trainer_v2.dev_runner.train_classification_model import ModelConfig
from trainer_v2.get_tpu_strategy import get_strategy
from trainer_v2.run_config import RunConfigEx


def load_weights2(some_layer, checkpoint_path):
    print('some_layer.weights', some_layer.weights)
    for t in some_layer.weights:
        print(t)
    checkpoint = tf.train.Checkpoint(encoder=some_layer)
    ret = checkpoint.read(checkpoint_path)
    ret.assert_existing_objects_matched()


def show_params(model_config, input_files, run_config):
    bert_config = model_config.bert_config
    bert_classifier: tf.keras.layers.Layer = BertClassifierLayer(bert_config,
                                                                 use_one_hot_embeddings=True,
                                                                 num_classes=model_config.num_classes,
                                                                 is_training=True)

    load_weights2(bert_classifier.bert_layer, run_config.init_checkpoint)
    exit()
    load_weights(bert_classifier.pooler, run_config.init_checkpoint)
    input_ids = tf.constant([101], tf.int32)
    table = bert_classifier.bert_layer.embedding_layer.embedding_table
    print(table.built)

    # Define model
    run_config.train_step = 100000
    max_seq_length = model_config.max_seq_length
    # model: tf.keras.models.Model = tf.keras.models.Model(inputs=inputs, outputs=output)
    # optimizer = tf.optimizers.Adam(learning_rate=run_config.learning_rate)
    # metrics = [tf.keras.metrics.SparseCategoricalAccuracy('accuracy', dtype=tf.float32)]
    # loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # model.compile(optimizer=optimizer, loss=loss, metrics=metrics)


def parse_input_files(input_file_str):
    input_files = []
    for input_pattern in input_file_str.split(","):
        input_files.extend(tf.io.gfile.glob(input_pattern))
    return input_files


def main(args):
    c_log.info("main {}".format(args))
    strategy = get_strategy(args.use_tpu, args.tpu_name)
    input_files = parse_input_files(args.input_files)
    bert_config = BertConfig.from_json_file(get_bert_config_path())
    max_seq_length = 300
    run_config = RunConfigEx(model_save_path=args.output_dir,
                             init_checkpoint=args.init_checkpoint)
    model_config = ModelConfig(bert_config, max_seq_length)
    with strategy.scope():
        # run_eval(model_config, checkpoint_path, input_files, max_seq_length, run_config)
        show_params(model_config, input_files, run_config)


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)
