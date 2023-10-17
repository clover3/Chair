import sys

from cpath import output_path
from data_generator.tokenizer_wo_tf import get_tokenizer
from misc_lib import path_join
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.alignment.predict_table.old.predict_d_terms_unigram import \
    build_model_with_output_mapping, predict_d_terms_for_job
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.path_helper2 import get_mmp_train_corpus_config


def predict_d_terms_per_job_and_save_temp_old(fetch_align_probe_fn):
    model_save_path = sys.argv[1]
    job_no = int(sys.argv[2])
    model_name = sys.argv[3]
    save_dir_path = path_join(output_path, "msmarco", "passage",
                         f"candidate_building_{model_name}")

    def get_model():
        return build_model_with_output_mapping(model_save_path, fetch_align_probe_fn)

    predict_d_terms_per_job_and_save(get_model, job_no, save_dir_path)


def predict_d_terms_per_job_and_save_temp(load_model_fn):
    c_log.info("predict_d_terms_per_job_and_save_temp")
    model_save_path = sys.argv[1]
    job_no = int(sys.argv[2])
    model_name = sys.argv[3]
    save_dir_path = path_join(output_path, "msmarco", "passage",
                              f"candidate_building_{model_name}")

    def get_model():
        return load_model_fn(model_save_path)

    predict_d_terms_per_job_and_save(get_model, job_no, save_dir_path)


def predict_d_terms_per_job_and_save(get_model, job_no, save_dir_path):
    def get_term_pair_candidate_building_path(job_no):
        save_path = path_join(save_dir_path, f"{job_no}.jsonl")
        return save_path

    tokenizer = get_tokenizer()
    c_log.info("Job no: %d", job_no)
    config = get_mmp_train_corpus_config()
    f = open(config.frequent_q_terms, "r")
    freq_q_terms = [line.strip() for line in f]
    model = get_model()
    save_path = get_term_pair_candidate_building_path(job_no)
    n_per_job = 100
    st = n_per_job * job_no
    ed = st + n_per_job
    q_term_list = [freq_q_terms[i] for i in range(st, ed)]
    predict_d_terms_for_job(q_term_list, model, tokenizer, save_path)

