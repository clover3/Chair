from ptorch.pep.get_pep_nseg_scoring import get_pepn_score_fn, load_model_from_conf
from ptorch.splade_tree.c2_log import c2_log, reset_log_formatter
import hydra
from omegaconf import DictConfig
import os

from cpath import yconfig_dir_path
from misc_lib import path_join
from ptorch.splade_tree.utils.utils import get_initialize_config

CONFIG_NAME = "pep_toy"
CONFIG_PATH = path_join(yconfig_dir_path, "hconfig")


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(exp_dict: DictConfig):
    reset_log_formatter()
    c2_log.info(__file__)
    exp_dict, config, init_dict, _ = get_initialize_config(exp_dict, train=True)
    model = load_model_from_conf(config, init_dict)
    score_fn = get_pepn_score_fn(config, init_dict)

    qd_list = [
        ("Where is the capital of France?", "The capital of France is Paris."),
        ("Where is the capital of France?", "The flag of the United States has 50 stars, each representing one of the 50 states in the country.")
    ]

    scores = score_fn(qd_list)
    print(qd_list)
    print(scores)


if __name__ == "__main__":
    main()

