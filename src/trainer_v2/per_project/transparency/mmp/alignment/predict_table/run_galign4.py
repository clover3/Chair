
import os

from trainer_v2.per_project.transparency.mmp.alignment.predict_table.old.predict_d_terms_mmp_train import \
    predict_d_terms_per_job_and_save_temp_old

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def fetch_align_probe_emb_concat(outputs):
    # GAlign4
    scores = outputs['align_probe']["emb_concat"][:, 0]
    return scores


# @report_run3
def main():
    fetch_align_probe_fn = fetch_align_probe_emb_concat
    predict_d_terms_per_job_and_save_temp_old(fetch_align_probe_fn)


if __name__ == "__main__":
    main()