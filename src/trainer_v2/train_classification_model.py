import json

import official.nlp.bert.configs
import tensorflow as tf
from official.nlp import bert

from cpath import get_bert_config_path, get_bert_full_path


def get_strategy(FLAGS):
    if FLAGS.use_tpu:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=FLAGS.tpu_name)
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.TPUStrategy(resolver)
    else:
        strategy = tf.distribute.MirroredStrategy()
    print("All devices: ", tf.config.list_logical_devices('TPU'))
    return strategy


def build_network(bert_config_file):
    print(2)
    config_dict = json.loads(tf.io.gfile.GFile(bert_config_file).read())
    print(3)
    bert_config = bert.configs.BertConfig.from_dict(config_dict)
    bert_classifier, bert_encoder = bert.bert_models.classifier_model(bert_config, num_labels=2)
    return bert_classifier, bert_encoder


def load_weights(bert_encoder, checkpoint_path):
    checkpoint = tf.train.Checkpoint(encoder=bert_encoder)
    checkpoint.read(checkpoint_path).assert_consumed()


def main():
    print("main1")
    checkpoint_path = get_bert_full_path()
    config_path = get_bert_config_path()
    bert_classifier, bert_encoder = build_network(config_path)
    print('bert_classifier', bert_classifier)
    print('bert_encoder', bert_encoder)
    load_weights(bert_encoder, checkpoint_path)


if __name__ == "__main__":
    main()
