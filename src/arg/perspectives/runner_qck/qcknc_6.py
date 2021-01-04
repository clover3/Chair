from arg.perspectives.ppnc.qck_job_starter import start_generate_jobs_for_dev
from arg.perspectives.qck.qcknc_datagen import get_eval_candidates_as_qck, is_correct_factory
from arg.qck.instance_generator.qcknc_datagen import QCKInstanceGenerator


def main():
    generator = QCKInstanceGenerator(get_eval_candidates_as_qck("dev"),
                                                           is_correct_factory())
    start_generate_jobs_for_dev(generator, "qcknc6")


if __name__ == "__main__":
    main()
