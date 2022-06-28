import tensorflow as tf

from taskman_client.task_proxy import get_local_machine_name
from trainer_v2.chair_logging import c_log


def get_strategy(use_tpu, tpu_name=None):
    if use_tpu:
        strategy = get_tpu_strategy_inner(tpu_name)
        # tpu_devices = tf.config.list_logical_devices('TPU')
        # c_log.info("{} TPU devices found".format(len(tpu_devices)))
    else:
        c_log.debug("use_tpu={}".format(use_tpu))
        strategy = tf.distribute.MultiWorkerMirroredStrategy()
        c_log.info("All devices: {}".format(tf.config.list_logical_devices('GPU')))
        # strategy = tf.distribute.MirroredStrategy()
    return strategy


def get_tpu_strategy_inner(tpu_name):
    from cloud_tpu_client import Client
    c_log.debug("get_tpu_strategy:: init TPUClusterResolver")
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu_name)
    c_log.debug("get_tpu_strategy:: experimental_connect_to_cluster")
    tf.config.experimental_connect_to_cluster(resolver)
    c_log.debug("get_tpu_strategy:: initialize_tpu_system")
    tf.tpu.experimental.initialize_tpu_system(resolver)
    c_log.debug("get_tpu_strategy:: init TPUStrategy")
    strategy = tf.distribute.TPUStrategy(resolver)
    c_log.debug("get_tpu_strategy:: init Client")
    c = Client(tpu=tpu_name)
    # c.configure_tpu_version(tf.__version__, restart_type='ifNeeded')
    return strategy


def get_strategy_by_machine_name():
    machine_name = get_local_machine_name()
    is_tpu = machine_name not in ["GOSFORD", "ingham.cs.umass.edu"]
    if is_tpu:
        strategy = get_strategy(True, "local")
    else:
        strategy = get_strategy(False, "")
    return strategy