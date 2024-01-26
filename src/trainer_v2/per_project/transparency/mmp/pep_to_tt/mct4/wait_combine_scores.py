import sys
from taskman_client.wait_job_group import wait_job_group
from taskman_client.wrapper3 import JobContext
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.pep_to_tt.mct4.filter_combine_scores import run_combination_filtering
from trainer_v2.per_project.transparency.mmp.pep_to_tt.mct4.score_mct4 import build_mct4_config


def main():
    model_name = sys.argv[1]
    step = int(sys.argv[2])
    conf = build_mct4_config(model_name, step)
    task_name = conf.job_name_base + "_combine"

    with JobContext(task_name):
        c_log.info("Waiting for job group %s ")
        wait_job_group(conf.job_name_base)
        c_log.info("Waiting Done")
        run_combination_filtering(conf)


if __name__ == "__main__":
    main()
