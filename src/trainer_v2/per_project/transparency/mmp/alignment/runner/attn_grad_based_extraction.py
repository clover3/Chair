import sys
import tensorflow as tf

from scratch.code2023.omegaconf_prac import PUConfig, ProcessingUnit, load_tf_predict_config
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.alignment.alignment_predictor import compute_alignment_any_pair
from trainer_v2.per_project.transparency.mmp.alignment.grad_extractor import extract_save_align
from omegaconf import OmegaConf
from cpath import output_path, data_path
from misc_lib import path_join
from trainer_v2.per_project.transparency.mmp.dev_analysis.when_term_frequency import enum_when_corpus
from trainer_v2.train_util.get_tpu_strategy import get_tpu_strategy_inner, device_list_summary
import atexit


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


def main():
    config = load_tf_predict_config(sys.argv[1])
    compute_alignment_fn = compute_alignment_any_pair
    def enum_qd_pairs():
        for query, doc_pos, doc_neg in enum_when_corpus():
            yield query, doc_pos
            yield query, doc_neg

    num_record = 13220
    qd_itr = enum_qd_pairs()
    save_path = config.save_path
    strategy = strategy_with_pu_config(config.pu_config)
    extract_save_align(compute_alignment_fn, qd_itr, strategy, save_path,
                       num_record, config.model_load_config.model_path,
                       config.batch_size)


if __name__ == "__main__":
    main()

