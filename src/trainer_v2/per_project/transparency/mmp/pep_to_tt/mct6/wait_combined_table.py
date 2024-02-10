import sys

from taskman_client.wait_task import wait_task
from taskman_client.wrapper3 import JobContext
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.pep_to_tt.mct6.score_mct6 import build_mct6_config
from trainer_v2.per_project.transparency.mmp.retrieval_run.bm25t_luk import run_bm25t_luk


def main():
    model_name = sys.argv[1]
    step = int(sys.argv[2])
    conf = build_mct6_config(model_name, step)
    combine_task_name = conf.job_name_base + "_combine"

    task_name = "wait_for_" + combine_task_name
    with JobContext(task_name):
        c_log.info("Waiting for task %s ", combine_task_name)
        wait_task(combine_task_name)
        c_log.info("Wait done")

    table_path = conf.table_save_path
    step_k = step // 1000
    run_name = f"mct6_{model_name}_{step_k}K"
    dataset = "trec_dl19"

    run_bm25t_luk(run_name, table_path, dataset)


if __name__ == "__main__":
    main()
