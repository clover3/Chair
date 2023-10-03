
import os
import sys

from cpath import output_path
from misc_lib import path_join
from taskman_client.wrapper3 import JobContext
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.alignment.predict_table.predict_d_terms_mmp_train import \
    predict_d_terms_per_job_and_save
from trainer_v2.train_util.get_tpu_strategy import get_strategy

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def fetch_align_probe_fn(outputs):
    # GAlign4
    scores = outputs['align_probe']["align_pred"][:, 0]
    return scores


def main():
    strategy = get_strategy()
    with strategy.scope():
        c_log.info(__file__)
        model_save_path = sys.argv[1]
        job_no = int(sys.argv[2])
        model_name = sys.argv[3]
        dir_path = path_join(output_path, "msmarco", "passage",
                             f"candidate_building_{model_name}")
        job_name = f"{model_name}_inf_{job_no}"
        with JobContext(job_name):
            predict_d_terms_per_job_and_save(fetch_align_probe_fn, job_no, dir_path, model_save_path)


if __name__ == "__main__":
    main()