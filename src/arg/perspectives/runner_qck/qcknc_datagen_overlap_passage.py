from arg.perspectives.runner_qck.qcknc_common import start_generate_jobs_for_val
from arg.perspectives.runner_qck.qcknc_datagen import get_eval_candidates_as_qck, is_correct_factory
from arg.qck.qcknc_datagen import QCKInstanceGenerator


def main():
    generator = QCKInstanceGenerator(get_eval_candidates_as_qck("train"), is_correct_factory())
    # Selected from doc_scorer_summarizer.py
    qk_candidate_name = "perspective_qk_candidate_train_dense"
    start_generate_jobs_for_val(generator, qk_candidate_name, "qcknc_dense")


if __name__ == "__main__":
    main()

