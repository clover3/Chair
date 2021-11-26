import sys

import keras
import official.nlp.bert.run_classifier
import official.nlp.optimization
import tensorflow as tf
from official import nlp
from official.nlp import bert

from cpath import get_bert_config_path
from models.keras_model.dev.from_hub import load_bert_model_by_hub
from tlm.model.base import BertConfig
from trainer_v2.arg_flags import flags_parser
from trainer_v2.chair_logging import c_log
from trainer_v2.get_tpu_strategy import get_strategy
from trainer_v2.run_config import RunConfigEx

keras = tf.keras


class ModelConfig:
    num_classes = 3
    max_seq_length = 512

    def __init__(self, bert_config, max_seq_length):
        self.bert_config = bert_config
        self.max_seq_length = max_seq_length


class BERT_CLS:
    def __init__(self, config: ModelConfig):
        num_classes = config.num_classes
        model, pooled_output, sequence_output = load_bert_model_by_hub(config.max_seq_length)
        self.seq_out = sequence_output
        output = tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax)(pooled_output)
        self.model: keras.Model = model
        self.pooler = pooled_output
        self.output = output


def get_optimizer(run_config: RunConfigEx):
    num_train_steps = run_config.train_step
    warmup_steps = int(num_train_steps * 0.1)
    optimizer = nlp.optimization.create_optimizer(
        run_config.learning_rate, num_train_steps=num_train_steps, num_warmup_steps=warmup_steps)
    return optimizer


def run_train(model_config, input_files, run_config):
    # Define model
    run_config.train_step = 100000
    max_seq_length = model_config.max_seq_length
    bert_cls = BERT_CLS(model_config)
    model = bert_cls.model
    optimizer = get_optimizer(run_config)
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy('accuracy', dtype=tf.float32)]
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    dataset = bert.run_classifier.get_dataset_fn(input_files,
                                                 max_seq_length,
                                                 run_config.batch_size,
                                                 is_training=True)()
    model.summary()
    print("dataset", dataset)
    model.fit(dataset, epochs=1, steps_per_epoch=run_config.train_step)
    model.save_weights(run_config.model_save_path)

#
# def run_eval(model_config, checkpoint_path, input_files, max_seq_length, run_config):
#     bert_config = model_config.bert_config
#     inputs = define_bert_keras_inputs(max_seq_length)
#     bert_classifier: tf.keras.layers.Layer = BertClassifierLayer(bert_config, True, model_config.num_classes)
#     output = bert_classifier(inputs)
#     model: tf.keras.models.Model = tf.keras.models.Model(inputs=inputs, outputs=output)
#     trainer_v2.misc_common.load_weights(checkpoint_path)


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
    max_seq_length = 128
    run_config = RunConfigEx(model_save_path=args.output_dir,
                             init_checkpoint=args.init_checkpoint)
    model_config = ModelConfig(bert_config, max_seq_length)
    with strategy.scope():
        # run_eval(model_config, checkpoint_path, input_files, max_seq_length, run_config)
        run_train(model_config, input_files, run_config)


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)
