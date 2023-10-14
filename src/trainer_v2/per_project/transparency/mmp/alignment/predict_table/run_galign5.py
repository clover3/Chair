
import os

from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.alignment.predict_table.old.predict_d_terms_mmp_train import \
    predict_d_terms_per_job_and_save_temp_old
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
        predict_d_terms_per_job_and_save_temp_old(fetch_align_probe_fn)



if __name__ == "__main__":
    main()