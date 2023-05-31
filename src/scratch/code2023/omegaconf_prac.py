import pathlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Union

from cpath import output_path, data_path
from misc_lib import path_join
from omegaconf import OmegaConf


class ProcessingUnit(Enum):
    any = 0
    CPU = 1
    GPU = 2
    TPU = 3


@dataclass
class PUConfig:
    target_PU: ProcessingUnit = ProcessingUnit.any
    force_use_gpu: bool = False


@dataclass
class ModelLoadConfig:
    model_path: str
    n_param: int
    # model_path_list: list


@dataclass
class TFPredictConfig:
    input_path: str
    save_path: str
    model_load_config: ModelLoadConfig
    model_load_config_path: str
    pu_config: PUConfig = field(default_factory=lambda: PUConfig())
    input_size: int = 0
    batch_size: int = 16


def test_pu_config():
    config_path = path_join(data_path, "config_a", "pu_config.yaml")
    conf = OmegaConf.structured(PUConfig)
    conf.merge_with(OmegaConf.load(config_path))
    print(conf)
    print(OmegaConf.to_yaml(conf))
    check_pu_config(conf)


def check_pu_config(conf):
    if conf.force_use_gpu:
        print("force use gpu", type(conf.force_use_gpu))
    else:
        print("not force use gpu", type(conf.force_use_gpu))
    print('type(conf.force_use_gpu)', type(conf.force_use_gpu))
    print('type(conf.target_PU)', type(conf.target_PU))
    print(conf.target_PU)


def test_predict_config():
    config_path = path_join(data_path, "config_a", "predict_config.yaml")
    conf = load_tf_predict_config(config_path)
    print(conf)
    print(OmegaConf.to_yaml(conf))
    pu_config = conf.pu_config
    check_pu_config(pu_config)


def load_tf_predict_config(config_path) -> TFPredictConfig:
    conf = OmegaConf.structured(TFPredictConfig)
    conf.merge_with(OmegaConf.load(config_path))
    if 'model_load_config' not in conf:
        model_load_config_path = path_join(data_path, "config_a", conf.model_load_config_path + ".yaml")
        model_load_config = OmegaConf.structured(ModelLoadConfig)
        model_load_config.merge_with(OmegaConf.load(model_load_config_path))
        conf.model_load_config = model_load_config
    return conf


def main():
    test_predict_config()


if __name__ == "__main__":
    main()