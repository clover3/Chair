from arg.perspectives.ppnc.qck_job_starter import start_generate_jobs_for_train_val
from arg.perspectives.qck.qcknc_datagen import get_eval_candidates_as_qck, is_correct_factory
from arg.qck.qcknc_datagen import QCKInstanceGenerator


def main():
    generator = QCKInstanceGenerator(get_eval_candidates_as_qck("train"),
                                                           is_correct_factory())
    start_generate_jobs_for_train_val(generator, "qcknc")


if __name__ == "__main__":
    main()
