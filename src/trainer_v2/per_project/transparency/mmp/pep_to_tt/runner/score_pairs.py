import os
import sys

from omegaconf import OmegaConf

from misc_lib import path_join, ceil_divide
from taskman_client.job_group_proxy import SubJobContext
from taskman_client.wrapper3 import JobContext
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.misc_common import read_lines
from trainer_v2.per_project.transparency.mmp.pep.term_pair_common import predict_save_top_k
from trainer_v2.per_project.transparency.mmp.pep_to_tt.pep_tt_modeling import PEP_TT_ModelConfig
from trainer_v2.per_project.transparency.mmp.pep_to_tt.inf_helper import PEP_TT_Inference
from trainer_v2.train_util.get_tpu_strategy import get_strategy2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'


def main():
    c_log.info(__file__)
    conf_path = sys.argv[1]
    slurm_job_idx = int(sys.argv[2])
    conf = OmegaConf.load(conf_path)
    q_terms = read_lines(conf.q_term_path)
    d_terms = read_lines(conf.d_term_path)
    job_size = int(conf.job_size)
    job_group_name = conf.run_name
    model_path = conf.model_path
    save_dir = conf.score_save_dir
    strategy = get_strategy2(False, force_use_gpu=True)
    with strategy.scope():
        model_config = PEP_TT_ModelConfig()

        inf_helper = PEP_TT_Inference(
            model_config,
            model_path,
            model_type=conf.model_type)

        num_items = len(q_terms)
        num_job_per_slurm_job = job_size
        num_jobs = ceil_divide(num_items, num_job_per_slurm_job)
        st = slurm_job_idx * job_size
        ed = st + job_size

        job_name = f"{job_group_name}_{slurm_job_idx}"
        with SubJobContext(job_group_name, slurm_job_idx, num_jobs):
            c_log.info("%s", job_name)
            for q_term_i in range(st, ed):
                c_log.info("q_term_i=%d", q_term_i)
                log_path = path_join(save_dir, f"{q_term_i}.txt")
                predict_term_pairs_fn = inf_helper.score_fn
                predict_save_top_k(
                    predict_term_pairs_fn, q_terms[q_term_i], d_terms,
                    log_path, outer_batch_size=100)

        c_log.info("Main terminate")


if __name__ == "__main__":
    main()