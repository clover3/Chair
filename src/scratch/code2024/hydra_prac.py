import os

import hydra
from omegaconf import DictConfig, OmegaConf

from cpath import yconfig_dir_path

print(yconfig_dir_path)

config_path = os.path.join(yconfig_dir_path, "experiment_confs/rerank_bm25t4")
config_name = "bm25t_empty"

print(config_name)


@hydra.main(config_path=yconfig_dir_path, config_name=config_name)
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    my_app()
