

from arg.perspectives.qck.qcknc_datagen import get_eval_candidates_as_qck, is_correct_factory
from arg.perspectives.runner_qck.qcknc_common import start_generate_jobs_for_sub_split
from arg.qck.qck_multi import MultiDocInstanceGenerator
from exec_lib import run_func_with_config


def main(config):
    job_name = "qck_multi"
    is_correct_fn = is_correct_factory()
    qk_candidate_name = "qk_candidate_msmarco_filtered_dev"
    generator = MultiDocInstanceGenerator(get_eval_candidates_as_qck("dev"), is_correct_fn, config)
    start_generate_jobs_for_sub_split(generator, qk_candidate_name, job_name, "dev")

    qk_candidate_name = "qk_candidate_msmarco_filtered_train"
    generator = MultiDocInstanceGenerator(get_eval_candidates_as_qck("train"), is_correct_fn, config)
    start_generate_jobs_for_sub_split(generator, qk_candidate_name, job_name, "train")
    generator = MultiDocInstanceGenerator(get_eval_candidates_as_qck("train"), is_correct_fn, config)
    start_generate_jobs_for_sub_split(generator, qk_candidate_name, job_name, "val")


if __name__ == "__main__":
    run_func_with_config(main)

