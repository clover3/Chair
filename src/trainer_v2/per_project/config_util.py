import atexit

import tensorflow as tf
from omegaconf import OmegaConf

from cpath import data_path
from misc_lib import path_join
from scratch.code2023.omegaconf_prac import PUConfig, ProcessingUnit
from trainer_v2.chair_logging import c_log
from trainer_v2.train_util.get_tpu_strategy import get_tpu_strategy_inner, device_list_summary


def load_pu_config():
    config_path = path_join(data_path, "config_a", "pu_config.yaml")
    conf = OmegaConf.structured(PUConfig)
    conf.merge_with(OmegaConf.load(config_path))
    return conf


def strategy_with_pu_config(pu_config):
    if pu_config.target_PU == ProcessingUnit.TPU:
        tpu_name = pu_config.tpu_name
        strategy = get_tpu_strategy_inner(tpu_name)
    elif pu_config.target_PU in \
            [ProcessingUnit.GPU, ProcessingUnit.any, ProcessingUnit.CPU]:
        strategy = tf.distribute.MultiWorkerMirroredStrategy()
        gpu_devices = tf.config.list_logical_devices('GPU')
        force_use_gpu = pu_config.get('force_use_gpu', False)
        if force_use_gpu and not gpu_devices:
            raise Exception("GPU devices not found")

        if pu_config.target_PU == ProcessingUnit.CPU and gpu_devices:
            c_log.warn("CPU was specified as target PU but GPU is actually used")
        c_log.info(device_list_summary(gpu_devices))
        try:
            atexit.register(strategy._extended._cross_device_ops._pool.close)  # type: ignore
            atexit.register(strategy._extended._host_cross_device_ops._pool.close)  # type: ignore
        except AttributeError:
            pass
    else:
        raise ValueError()

    return strategy


def get_strategy_with_default_pu_config():
    return strategy_with_pu_config(load_pu_config())