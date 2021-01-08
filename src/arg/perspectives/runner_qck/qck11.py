from arg.perspectives.ppnc.qck_job_starter import start_generate_jobs
from arg.perspectives.qck.qcknc_datagen import get_eval_candidates_as_qck, is_correct_factory
from arg.qck.instance_generator.qcknc_datagen import QCKInstanceGenerator


def main():
    for split in ["dev", "test"]:
        generator = QCKInstanceGenerator(get_eval_candidates_as_qck(split), is_correct_factory())
        # Selected from doc_scorer_summarizer.py
        qk_candidate_name = "pc_qk2_{}_cpnc12_filtered".format(split)
        start_generate_jobs(generator, split, qk_candidate_name, "qck11")


if __name__ == "__main__":
    main()

