import logging
import sys

from contradiction.esnli.runner.run_ts import get_todo
from contradiction.medical_claims.token_tagging.solvers.search_solver import PartialSegSolver, WordSegSolver
from contradiction.solve_run_helper import solve_esnli_tag
from data_generator.NLI.enlidef import get_target_class, enli_tags
from epath import job_man_dir
from job_manager.job_runner_with_server import JobRunnerF
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.definitions import ModelConfig300_3
from trainer_v2.custom_loop.run_config2 import get_eval_run_config2
from trainer_v2.custom_loop.train_loop_helper import get_strategy_from_config
from trainer_v2.keras_server.bert_like_client import BERTClientCore
from trainer_v2.keras_server.bert_like_server import get_keras_bert_like_predict_fn
from trainer_v2.train_util.arg_flags import flags_parser


def get_predictor(args):
    model_config = ModelConfig300_3()
    run_config = get_eval_run_config2(args)
    strategy = get_strategy_from_config(run_config)
    model_path = run_config.get_model_path()
    predict = get_keras_bert_like_predict_fn(model_path, model_config, strategy)
    client = BERTClientCore(predict, model_config.max_seq_length)
    return client.request_multiple


def do_for_label(predict_fn, tag_type, split):
    target_idx = get_target_class(tag_type)
    solver = WordSegSolver(target_idx, predict_fn)
    run_name = "token_entail"
    solve_esnli_tag(split, run_name, solver, tag_type)


def main(args):
    c_log.setLevel(logging.WARN)
    todo_list = get_todo()
    predict_fn = get_predictor(args)
    def work_fn(job_id):
        tag_type, split = todo_list[job_id]
        do_for_label(predict_fn, tag_type, split)

    num_jobs = len(todo_list)
    runner = JobRunnerF(job_man_dir, num_jobs, "esnli_snli7", work_fn)
    runner.auto_runner()


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)
