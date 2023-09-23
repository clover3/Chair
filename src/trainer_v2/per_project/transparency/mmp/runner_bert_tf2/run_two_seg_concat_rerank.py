import sys
from taskman_client.task_proxy import get_task_manager_proxy
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.eval_helper.eval_line_format import eval_dev_mrr, predict_and_batch_save_scores
from trainer_v2.per_project.transparency.mmp.eval_helper.pep_rerank import get_pep_scorer
from trainer_v2.train_util.arg_flags import flags_parser
from trainer_v2.train_util.get_tpu_strategy import get_strategy
from typing import List, Iterable, Callable, Dict, Tuple, Set


def main(args):
    model_path = args.output_dir
    run_name = args.run_name
    dataset = "dev_sample100"
    c_log.info(f"{run_name}, {dataset}")
    strategy = get_strategy(args.use_tpu, args.tpu_name)
    with strategy.scope():
        c_log.info("Building scorer")
        score_fn: Callable[[List[Tuple[str, str]]], Iterable[float]] = get_pep_scorer(model_path)
    predict_and_batch_save_scores(score_fn, dataset, run_name, 100*100)
    score = eval_dev_mrr(dataset, run_name)
    proxy = get_task_manager_proxy()
    proxy.report_number(run_name, score, dataset, "mrr")
    print(f"Recip_rank:\t{score}")


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)
