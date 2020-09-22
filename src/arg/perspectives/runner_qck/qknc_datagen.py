from arg.perspectives.ppnc.qck_job_starter import start_generate_jobs_for_train_val
from arg.qck.qknc_datagen import QKInstanceGenerator

# Make payload without any annotation

def main():
    def is_correct(dummy_query, dummy_passage):
        return 0

    start_generate_jobs_for_train_val(QKInstanceGenerator(is_correct), "qknc")


if __name__ == "__main__":
    main()