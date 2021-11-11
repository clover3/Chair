import sys

import keras
import tensorflow as tf

import trainer_v2.checkpoint_util
from cpath import get_bert_config_path
from models.keras_model.bert_keras.modular_bert import BertClassifierLayer, define_bert_keras_inputs
from tlm.model.base import BertConfig
from trainer_v2.arg_flags import flags_parser
from trainer_v2.chair_logging import c_log
from trainer_v2.checkpoint_util import load_weights
from trainer_v2.get_tfrecord_dataset import get_classification_dataset
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
    def __init__(self, bert_params, config: ModelConfig):
        import bert
        l_bert = bert.BertModelLayer.from_params(bert_params, name="bert")
        max_seq_len = config.max_seq_length
        num_classes = config.num_classes

        l_input_ids = keras.layers.Input(shape=(max_seq_len,), dtype='int32', name="input_ids")
        l_token_type_ids = keras.layers.Input(shape=(max_seq_len,), dtype='int32', name="segment_ids")
        seq_out = l_bert([l_input_ids, l_token_type_ids])  # [batch_size, max_seq_len, hidden_size]
        self.seq_out = seq_out
        first_token = seq_out[:, 0, :]
        pooler = tf.keras.layers.Dense(bert_params.hidden_size, activation=tf.nn.tanh, name="bert/pooler/dense")
        pooled = pooler(first_token)
        output = tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax)(pooled)
        model = keras.Model(inputs=[l_input_ids, l_token_type_ids], outputs=output, name="bert_model")
        self.model: keras.Model = model
        self.l_bert = l_bert
        self.pooler = pooler
        self.output = output


def run_train(model_config, input_files, run_config):
    bert_config = model_config.bert_config
    bert_classifier: tf.keras.layers.Layer = BertClassifierLayer(bert_config,
                                                                 use_one_hot_embeddings=True,
                                                                 num_classes=model_config.num_classes,
                                                                 is_training=True)

    load_weights(bert_classifier.bert_layer, run_config.init_checkpoint)
    load_weights(bert_classifier.pooler, run_config.init_checkpoint)
    # Define model
    run_config.train_step = 100000
    max_seq_length = model_config.max_seq_length
    inputs = define_bert_keras_inputs(max_seq_length)
    output = bert_classifier(inputs)
    model: tf.keras.models.Model = tf.keras.models.Model(inputs=inputs, outputs=output)
    optimizer = tf.optimizers.Adam(learning_rate=run_config.learning_rate)
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy('accuracy', dtype=tf.float32)]
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    dataset = get_classification_dataset(input_files, max_seq_length, True, run_config.batch_size)
    # c_log.info("Dataset length={}".format(len(dataset)))
    model.summary()
    model.fit(dataset, epochs=1)
    model.save_weights(run_config.model_save_path)


def run_eval(model_config, checkpoint_path, input_files, max_seq_length, run_config):
    bert_config = model_config.bert_config
    inputs = define_bert_keras_inputs(max_seq_length)
    bert_classifier: tf.keras.layers.Layer = BertClassifierLayer(bert_config, True, model_config.num_classes)
    output = bert_classifier(inputs)
    model: tf.keras.models.Model = tf.keras.models.Model(inputs=inputs, outputs=output)
    trainer_v2.misc_common.load_weights(checkpoint_path)


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
        run_train(model_config, input_files, run_config)


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)
