import sys
from omegaconf import OmegaConf

from misc_lib import path_join, TimeEstimator
from taskman_client.wrapper3 import JobContext
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.misc_common import read_lines, load_first_column
from trainer_v2.per_project.transparency.mmp.pep.term_pair_common import predict_save_top_k
from trainer_v2.per_project.transparency.mmp.pep_to_tt.inf_helper import PEP_TT_Inference
from trainer_v2.per_project.transparency.mmp.pep_to_tt.pep_tt_modeling import PEP_TT_ModelConfig


def score_given_pair_from_conf(slurm_job_idx, conf):
    c_log.info("score_given_pair_from_conf. slurm_job_idx=%d", slurm_job_idx)
    model_path = conf.model_path
    job_name_base = conf.job_name_base
    save_dir = conf.score_save_dir
    job_size = int(conf.job_size)
    q_terms = read_lines(conf.q_term_path)
    model_config = PEP_TT_ModelConfig()
    inf_helper = PEP_TT_Inference(
        model_config,
        model_path,
        model_type=conf.model_type)
    st = slurm_job_idx * job_size
    ed = min(st + job_size, len(q_terms))
    job_name = f"{job_name_base}_{slurm_job_idx}"
    with JobContext(job_name):
        predict_term_pairs_fn = inf_helper.score_fn
        n_iter = ed - st
        ticker = TimeEstimator(n_iter)
        for q_term_i in range(st, ed):
            c_log.info(f"Term {q_term_i}")
            target_term_table_path = path_join(conf.d_term_dir, f"{q_term_i}.txt")
            log_path = path_join(save_dir, f"{q_term_i}.txt")
            try:
                d_terms = load_first_column(target_term_table_path)
                predict_save_top_k(
                    predict_term_pairs_fn, q_terms[q_term_i], d_terms,
                    log_path, outer_batch_size=100, verbose=False)
            except FileNotFoundError:
                c_log.warn(f"Term {q_term_i} is not found")
                pass

            ticker.tick()


def main():
    conf_path = sys.argv[1]
    conf = OmegaConf.load(conf_path)
    slurm_job_idx = int(sys.argv[2])
    score_given_pair_from_conf(slurm_job_idx, conf)


if __name__ == "__main__":
    main()
