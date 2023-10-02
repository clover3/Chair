import os
import sys

import yaml

from data_generator.job_runner import WorkerInterface
from table_lib import tsv_iter
from job_manager.job_runner_with_server import JobRunnerS
from misc_lib import select_third_fourth, path_join
from taskman_client.task_proxy import get_task_manager_proxy
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.eval_helper.mmp_eval_line_format import eval_dev_mrr, \
    predict_and_batch_save_scores
from trainer_v2.per_project.transparency.mmp.eval_helper.eval_line_format import score_and_save_score_lines
from trainer_v2.per_project.transparency.mmp.eval_helper.rerank import get_scorer
from trainer_v2.train_util.arg_flags import flags_parser
from trainer_v2.train_util.get_tpu_strategy import get_strategy
from typing import List, Iterable, Callable, Dict, Tuple, Set


class PredWorker(WorkerInterface):
    def __init__(self, config, output_dir):
        self.output_dir = output_dir
        use_tpu = "tpu_name" in config
        if not "tpu_name" in config:
            config["tpu_name"] = ""
        self.quad_tsv_dir = config['quad_tsv_dir']
        model_path = config['model_path']
        self.scores_path = config['score_save_path']

        strategy = get_strategy(use_tpu, config['tpu_name'])
        with strategy.scope():
            c_log.info("Building scorer")
            self.score_fn = get_scorer(model_path)

    def work(self, job_id):
        quad_tsv_path = os.path.join(self.quad_tsv_dir, str(job_id))
        tuple_itr: Iterable[Tuple[str, str]] = select_third_fourth(tsv_iter(quad_tsv_path))
        max_batch_size = 1024
        data_size = 1000 * 10000
        score_and_save_score_lines(
            tuple_itr, self.score_fn, self.scores_path, max_batch_size, data_size)


def main():
    with open(sys.argv[1], 'r') as file:
        config = yaml.safe_load(file)

    def worker_factory(output_dir):
        return PredWorker(
            config, output_dir
        )

    working_dir = "local_data"
    num_job = 13
    job_name = "mmp_train_pred"
    runner = JobRunnerS(working_dir, num_job, job_name, worker_factory)
    runner.auto_runner()


if __name__ == "__main__":
    main()

