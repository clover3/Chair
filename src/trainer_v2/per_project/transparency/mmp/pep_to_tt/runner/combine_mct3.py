import sys

from omegaconf import OmegaConf

from trainer_v2.per_project.transparency.mmp.pep.term_pair_scoring_runner.combination_filtering_constant import \
    run_combination_filtering
from trainer_v2.per_project.transparency.mmp.pep_to_tt.runner.score_mct3 import build_mct3_config


def main():
    model_name = sys.argv[1]
    step = sys.argv[2]

    # Build config
    conf = build_mct3_config(model_name, step)
    # Convert to OmegaConf
    omega_conf = OmegaConf.create(conf)
    run_combination_filtering(omega_conf)


if __name__ == "__main__":
    main()
