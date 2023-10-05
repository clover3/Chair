import sys

from cpath import output_path
from data_generator.tokenizer_wo_tf import get_tokenizer
from misc_lib import path_join
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.alignment.predict_table.predict_d_terms_unigram import \
    build_model_with_output_mapping, predict_d_terms
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.path_helper2 import get_mmp_train_corpus_config


def predict_d_terms_per_job_and_save_temp(fetch_align_probe_fn):
    c_log.info(__file__)
    model_save_path = sys.argv[1]
    job_no = int(sys.argv[2])
    model_name = sys.argv[3]
    save_dir_path = path_join(output_path, "msmarco", "passage",
                         f"candidate_building_{model_name}")

    predict_d_terms_per_job_and_save(fetch_align_probe_fn, job_no, save_dir_path, model_save_path)


def predict_d_terms_per_job_and_save(fetch_align_probe_fn, job_no, save_dir_path, model_save_path):
    def get_term_pair_candidate_building_path(job_no):
        save_path = path_join(save_dir_path, f"{job_no}.jsonl")
        return save_path

    tokenizer = get_tokenizer()
    c_log.info("Job no: %d", job_no)
    config = get_mmp_train_corpus_config()
    f = open(config.frequent_q_terms, "r")
    freq_q_terms = [line.strip() for line in f]
    model = build_model_with_output_mapping(model_save_path, fetch_align_probe_fn)
    save_path = get_term_pair_candidate_building_path(job_no)
    predict_d_terms(freq_q_terms, model, tokenizer, save_path, job_no)