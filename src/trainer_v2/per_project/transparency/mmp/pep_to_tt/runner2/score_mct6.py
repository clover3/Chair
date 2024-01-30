import logging
import sys
from omegaconf import OmegaConf

from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.pep_to_tt.runner.score_given_pairs2 import score_given_pair_from_conf
from trainer_v2.train_util.get_tpu_strategy import get_strategy2


def build_mct6_config(model_name, step):
    # Hard-coded configuration with placeholders for model_name and step
    conf = {
        "model_name": model_name,
        "step": step,
        "model_step_name": f"{model_name}_{step}",
        "model_path": f"output/model/runs/{model_name}/model_{step}",
        "q_term_path": "output/mmp/lucene_krovetz/freq100K.txt",
        "d_term_dir": f"output/mmp/mmp_freq100K_pair_pep_tt17_10K",
        "job_size": 100,
        "job_name_base": f"mct6_{model_name}_{step}",
        "score_save_dir": f"output/mmp/mct6_{model_name}_{step}",
        "table_save_path": f"output/mmp/tables/mtc6_{model_name}_{step}.tsv",
        "constant_threshold": 0.1,
        "model_type": "PEP_TT_Model_Single"
    }

    return conf


def main():
    c_log.setLevel(logging.INFO)
    c_log.info(__file__)
    model_name = sys.argv[1]
    step = int(sys.argv[2])
    job_idx = int(sys.argv[3])

    # Build config
    conf = build_mct6_config(model_name, step)
    omega_conf = OmegaConf.create(conf)

    strategy = get_strategy2(False, force_use_gpu=True)
    with strategy.scope():
        score_given_pair_from_conf(job_idx, omega_conf)



if __name__ == "__main__":
    main()
