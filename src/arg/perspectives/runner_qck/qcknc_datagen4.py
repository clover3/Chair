from arg.perspectives.qck.qcknc_datagen import get_eval_candidates_as_qck, is_correct_factory
from arg.perspectives.runner_qck.qcknc_common import start_generate_jobs_for_train, start_generate_jobs_for_val, \
    start_generate_jobs_for_sub_split
from arg.qck.instance_generator.qcknc_datagen import QCKInstanceGenerator


def main():
    job_name = "qcknc4"
    qk_candidate_name = "perspective_qk_candidate_filtered_dev"
    generator = QCKInstanceGenerator(get_eval_candidates_as_qck("dev"), is_correct_factory())
    start_generate_jobs_for_sub_split(generator, qk_candidate_name, job_name, "dev")

    # Selected from doc_scorer_summarizer.py
    qk_candidate_name = "perspective_qk_candidate_filtered_train"
    generator = QCKInstanceGenerator(get_eval_candidates_as_qck("train"), is_correct_factory())
    start_generate_jobs_for_train(generator, qk_candidate_name, job_name)
    generator = QCKInstanceGenerator(get_eval_candidates_as_qck("train"), is_correct_factory())
    start_generate_jobs_for_val(generator, qk_candidate_name, job_name)


if __name__ == "__main__":
    main()

