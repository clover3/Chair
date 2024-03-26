import os
import sys

import hydra
from omegaconf import DictConfig
import logging
from cpath import yconfig_dir_path
from misc_lib import path_join

from ptorch.splade_tree.c2_log import c2_log, reset_log_formatter
from ptorch.splade_tree.utils.utils import get_initialize_config

CONFIG_NAME = os.environ["HCONFIG_NAME"]
CONFIG_PATH = path_join(yconfig_dir_path, "hconfig")


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def train(exp_dict: DictConfig):
    reset_log_formatter()
    c2_log.info("train train()")
    print("CONFIG_NAME", CONFIG_NAME)
    exp_dict, config, init_dict, _ = get_initialize_config(exp_dict, train=True)
    print(exp_dict)


if __name__ == "__main__":
    train()
