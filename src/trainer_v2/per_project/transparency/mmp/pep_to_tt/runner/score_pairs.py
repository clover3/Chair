import sys

from omegaconf import OmegaConf

from misc_lib import path_join
from taskman_client.job_group_proxy import SubJobContext
from trainer_v2.per_project.transparency.misc_common import read_lines
from trainer_v2.per_project.transparency.mmp.pep.term_pair_common import predict_save_top_k
from trainer_v2.per_project.transparency.mmp.pep_to_tt.pep_tt_modeling import PEP_TT_ModelConfig
from trainer_v2.per_project.transparency.mmp.pep_to_tt.runner.interactive_inference import PEP_TT_Inference


def main():
    conf_path = sys.argv[1]
    slurm_job_idx = int(sys.argv[2])
    conf = OmegaConf.load(conf_path)
    q_terms = read_lines(conf.q_term_path)
    d_terms = read_lines(conf.d_term_path)
    job_size = int(conf.job_size)
    job_name = conf.run_name
    model_path = conf.model_path
    save_dir = conf.score_save_dir


    model_config = PEP_TT_ModelConfig()
    inf_helper = PEP_TT_Inference(model_config, model_path)
    num_items = len(q_terms)
    max_job = num_items

    num_job_per_slurm_job = job_size
    num_slurm_job = max_job // num_job_per_slurm_job + 1
    st = slurm_job_idx * job_size
    ed = st + job_size

    for q_term_i in range(st, ed):
        log_path = path_join(save_dir, f"{q_term_i}.txt")
        predict_term_pairs_fn = inf_helper.score_fn
        with SubJobContext(job_name, q_term_i, max_job):
            predict_save_top_k(
                predict_term_pairs_fn, q_terms[q_term_i], d_terms,
                log_path, outer_batch_size=100)


if __name__ == "__main__":
    main()