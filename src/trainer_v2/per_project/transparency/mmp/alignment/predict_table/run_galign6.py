
import os

from trainer_v2.per_project.transparency.mmp.alignment.predict_table.predict_d_terms_mmp_train import \
    predict_d_terms_per_job_and_save

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def fetch_align_probe_g_attention_output(outputs):
    # GAlign6
    scores = outputs['align_probe']["g_attention_output"][:, 0]
    return scores


# @report_run3
def main():
    fetch_align_probe_fn = fetch_align_probe_g_attention_output
    predict_d_terms_per_job_and_save(fetch_align_probe_fn)


if __name__ == "__main__":
    main()