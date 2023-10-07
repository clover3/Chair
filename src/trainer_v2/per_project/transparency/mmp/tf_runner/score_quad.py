import sys

import yaml

from table_lib import tsv_iter
from misc_lib import select_third_fourth
from taskman_client.task_proxy import get_task_manager_proxy
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.eval_helper.mmp_eval_line_format import eval_dev_mrr, \
    predict_and_batch_save_scores
from trainer_v2.per_project.transparency.mmp.eval_helper.eval_line_format import batch_score_and_save_score_lines
from trainer_v2.per_project.transparency.mmp.eval_helper.rerank import get_scorer
from trainer_v2.train_util.arg_flags import flags_parser
from trainer_v2.train_util.get_tpu_strategy import get_strategy
from typing import List, Iterable, Callable, Dict, Tuple, Set


def main():
    with open(sys.argv[1], 'r') as file:
        config = yaml.safe_load(file)

    quad_tsv_path = config['quad_tsv_path']
    model_path = config['model_path']
    scores_path = config['score_save_path']
    print(config)
    tuple_itr: Iterable[Tuple[str, str]] = select_third_fourth(tsv_iter(quad_tsv_path))
    max_batch_size = 1024
    data_size = 1000 * 10000
    use_tpu = "tpu_name" in config
    if not "tpu_name" in config:
        config["tpu_name"] = ""

    strategy = get_strategy(use_tpu, config['tpu_name'])
    with strategy.scope():
        c_log.info("Building scorer")
        score_fn = get_scorer(model_path)
        batch_score_and_save_score_lines(score_fn, tuple_itr, scores_path, data_size, max_batch_size)


if __name__ == "__main__":
    main()

