import tensorflow as tf

from trainer_v2.chair_logging import c_log


def get_strategy(use_tpu, tpu_name=None):
    if use_tpu:
        strategy = get_tpu_strategy_inner(tpu_name)
        tpu_devices = tf.config.list_logical_devices('TPU')
        c_log.info("{} TPU devices found".format(len(tpu_devices)))
    else:
        c_log.debug("use_tpu={}".format(use_tpu))
        strategy = tf.distribute.MultiWorkerMirroredStrategy()
        c_log.info("All devices: {}".format(tf.config.list_logical_devices('GPU')))
        # strategy = tf.distribute.MirroredStrategy()
    return strategy


def get_tpu_strategy_inner(tpu_name):
    from cloud_tpu_client import Client
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu_name)
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.TPUStrategy(resolver)
    c = Client(tpu=tpu_name)
    c.configure_tpu_version(tf.__version__, restart_type='ifNeeded')
    return strategy
