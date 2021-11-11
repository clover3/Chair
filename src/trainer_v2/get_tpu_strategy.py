import tensorflow as tf
from cloud_tpu_client import Client

from trainer_v2.chair_logging import c_log


def get_strategy(use_tpu, tpu_name=None):
    if use_tpu:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu_name)
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.TPUStrategy(resolver)
        c = Client(tpu=tpu_name)
        c.configure_tpu_version(tf.__version__, restart_type='ifNeeded')

    else:
        strategy = tf.distribute.MirroredStrategy()
    c_log.info("All devices: {}".format(tf.config.list_logical_devices('TPU')))
    return strategy