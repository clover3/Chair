import logging
import sys

from omegaconf import OmegaConf

from misc_lib import path_join
from table_lib import tsv_iter
from taskman_client.job_group_proxy import SubJobContext
from taskman_client.wrapper3 import JobContext
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.pep_to_tt.inf_helper import PEP_TT_Inference
from trainer_v2.per_project.transparency.mmp.pep_to_tt.pep_tt_common import predict_pairs_save
from trainer_v2.per_project.transparency.mmp.pep_to_tt.pep_tt_modeling import PEP_TT_ModelConfig
from trainer_v2.train_util.get_tpu_strategy import get_strategy2


def build_mct4_config(model_name, step):
    # Hard-coded configuration with placeholders for model_name and step
    conf = {
        "model_name": model_name,
        "step": step,
        "model_step_name": f"{model_name}_{step}",
        "model_path": f"output/model/runs/{model_name}/model_{step}",
        "src_term_pair_path": f"output/mmp/table4_01.tsv",
        "job_size": 100000,
        # 6,797,269 terms,
        # 68 jobs
        "num_jobs": 68,
        "job_name_base": f"mct4_{model_name}_{step}",
        "score_save_dir": f"output/mmp/mct4_{model_name}_{step}",
        "table_save_path": f"output/mmp/tables/mtc4_{model_name}_{step}.tsv",
        "constant_threshold": 0.1,
        "model_type": "PEP_TT_Model_Single"
    }

    return OmegaConf.create(conf)


def main():
    c_log.setLevel(logging.INFO)
    c_log.info(__file__)
    model_name = sys.argv[1]
    step = int(sys.argv[2])
    job_idx = int(sys.argv[3])

    # Build config
    conf = build_mct4_config(model_name, step)
    model_path = conf.model_path
    job_name_base = conf.job_name_base
    c_log.info("%s %d", job_name_base, job_idx)
    with SubJobContext(job_name_base, job_idx, 68):
        strategy = get_strategy2(False, force_use_gpu=True)
        with strategy.scope():
            model_config = PEP_TT_ModelConfig()
            inf_helper = PEP_TT_Inference(
                model_config,
                model_path,
                model_type=conf.model_type)

            all_term_pairs = [(qt, dt) for qt, dt, _score in tsv_iter(conf.src_term_pair_path)]
            st = job_idx * conf.job_size
            ed = min(st + conf.job_size, len(all_term_pairs))
            term_pair_itr = all_term_pairs[st:ed]

            save_path = path_join(conf.score_save_dir, f"{job_idx}.txt")
            predict_pairs_save(
                inf_helper.score_fn,
                term_pair_itr,
                save_path,
                1000
            )


if __name__ == "__main__":
    main()
