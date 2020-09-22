from arg.perspectives.runner_qck.qcknc_datagen import get_eval_candidates_as_qck, is_correct_factory
from arg.perspectives.runner_qck.qcknc_pred_datagen import run_jobs_with_qk_candidate
from arg.qck.qcknc_datagen import QCKInstanceGenerator


def main():
    sub_split = "dev"
    name_prefix = "qcknc2"
    qk_candidate_name = "qk_stage2_dev_2"

    generator = QCKInstanceGenerator(get_eval_candidates_as_qck(sub_split), is_correct_factory())
    run_jobs_with_qk_candidate(generator, sub_split, qk_candidate_name, name_prefix)


if __name__ == "__main__":
    main()