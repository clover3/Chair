from arg.perspectives.ppnc.qck_job_starter import start_generate_jobs_for_train
from arg.qck.decl import KDP, QCKQuery
# Make payload without any annotation
from arg.qck.qk_regression_datagen import QKRegressionInstanceGenerator


def main():
    max_score = 0.01
    min_score = -0.01

    def get_score(query: QCKQuery, passage: KDP):
        return 0.01

    start_generate_jobs_for_train(QKRegressionInstanceGenerator(get_score), "qk_regression")


if __name__ == "__main__":
    main()