import os.path
import sys
import time

from omegaconf import OmegaConf

from misc_lib import path_join, TimeEstimator
from table_lib import tsv_iter
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.pep.inf_helper import predict_term_pairs_and_save
from trainer_v2.per_project.transparency.mmp.pep_to_tt.inf_helper import PEP_TT_Inference
from trainer_v2.per_project.transparency.mmp.pep_to_tt.pep_tt_modeling import PEP_TT_ModelConfig

def wait_file(file_path):
    file_ready = False
    while not file_ready:
        file_ready = os.path.exists(file_path) and not os.path.getsize(file_path) == 0
        time.sleep(60 * 2)


def main():
    # confs/pred_align_cands.yaml
    conf_path = sys.argv[1]
    conf = OmegaConf.load(conf_path)

    slurm_job_no = int(sys.argv[2])
    model_config = PEP_TT_ModelConfig()
    inf_helper = PEP_TT_Inference(
        model_config,
        conf.model_path)

    st = slurm_job_no * 10
    ed = st + 10
    for i in range(st, ed):
        c_log.info(f"Job %d", i)
        job_no = i
        src_file_path = path_join(conf.candidate_dir, f"{job_no}.txt")
        score_save_path = path_join(conf.score_save_dir, f"{job_no}.txt")

        wait_file(src_file_path)
        if os.path.exists(score_save_path):
            c_log.info("Skip %d", i)
            continue

        n_line = 2000
        ticker = TimeEstimator(n_line)

        def enum_pairs():
            for row in tsv_iter(src_file_path):
                if row:
                    qt = row[0]
                    d_terms = row[1:]
                    for dt in d_terms:
                        yield qt, dt
                ticker.tick()

        predict_term_pairs_and_save(
            inf_helper.score_fn,
            enum_pairs(),
            score_save_path,
            1024,
            None)
        c_log.info("Done")


if __name__ == "__main__":
    main()
