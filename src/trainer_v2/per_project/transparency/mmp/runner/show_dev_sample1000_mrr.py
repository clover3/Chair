import logging
import sys

from taskman_client.task_proxy import get_task_manager_proxy
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.eval_helper.eval_line_format import eval_dev_mrr


def main():
    dataset = "dev_sample1000"
    run_name = sys.argv[1]
    c_log.setLevel(logging.DEBUG)
    metric = "mrr"
    score = eval_dev_mrr(dataset, run_name)
    print(f"{metric}:\t{score}")
    proxy = get_task_manager_proxy()
    proxy.report_number(run_name, score, dataset, metric)


if __name__ == "__main__":
    main()