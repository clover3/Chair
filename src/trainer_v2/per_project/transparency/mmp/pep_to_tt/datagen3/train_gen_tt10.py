import sys

from omegaconf import OmegaConf

from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.pep_to_tt.datagen3.train_gen import run_pep_tt_encoding_jobs
from trainer_v2.per_project.transparency.mmp.pep_to_tt.dataset_builder import \
    get_pep_tt_single_encoder2
from trainer_v2.per_project.transparency.mmp.pep_to_tt.pep_tt_modeling import PEP_TT_ModelConfig


def main():
    conf = OmegaConf.load(sys.argv[1])
    model_config = PEP_TT_ModelConfig()
    job_no = int(sys.argv[2])
    c_log.info("Job %d", job_no)
    encoder = get_pep_tt_single_encoder2(model_config, conf)
    run_pep_tt_encoding_jobs(encoder, conf, job_no)


if __name__ == "__main__":
    main()
