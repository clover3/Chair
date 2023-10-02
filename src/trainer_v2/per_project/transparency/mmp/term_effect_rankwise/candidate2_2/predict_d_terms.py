import os

from trainer_v2.per_project.transparency.mmp.alignment.predict_table.predict_d_terms_unigram import predict_d_terms, \
    build_model_with_output_mapping

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
from trainer_v2.chair_logging import c_log

from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.candidate2_2.path_helper import \
    get_candidate2_2_term_pair_candidate_building_path
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.path_helper2 import MMPGAlignPathHelper, get_cand2_2_path_helper
from data_generator.tokenizer_wo_tf import get_tokenizer


def fetch_align_probe_all_concat(outputs):
    scores = outputs['align_probe']['all_concat'][:, 0]
    return scores


# @report_run3
def main():
    c_log.info(__file__)
    model_save_path = sys.argv[1]
    job_no = int(sys.argv[2])

    tokenizer = get_tokenizer()
    print("Job no:", job_no)
    config: MMPGAlignPathHelper = get_cand2_2_path_helper()
    save_path = get_candidate2_2_term_pair_candidate_building_path(job_no)
    freq_q_terms = config.load_freq_q_terms()
    model = build_model_with_output_mapping(model_save_path, fetch_align_probe_all_concat)
    predict_d_terms(freq_q_terms, model, tokenizer, save_path, job_no)


if __name__ == "__main__":
    main()