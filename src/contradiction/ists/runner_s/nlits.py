import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from contradiction.ists.predict_common import eval_ists_noali_headlines_train_batch
from contradiction.medical_claims.token_tagging.solvers.nlits_batch_solver import get_batch_solver_nlits6, \
    SolverPostProcessorPunct
from taskman_client.wrapper3 import report_run3
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.run_config2 import get_eval_run_config2
from trainer_v2.train_util.arg_flags import flags_parser


@report_run3
def main(args):
    c_log.info("Start {}".format(__file__))
    run_config = get_eval_run_config2(args)
    run_name = "nlits"
    target_label = 1
    solver = get_batch_solver_nlits6(run_config, "concat", target_label)
    eval_ists_noali_headlines_train_batch(run_name, solver)


@report_run3
def solve_w_post(args):
    c_log.info("Start {}".format(__file__))
    run_config = get_eval_run_config2(args)
    target_label = 1
    solver = get_batch_solver_nlits6(run_config, "concat", target_label)
    run_name = "nlits_punc"
    c_log.info("run name= {}".format(run_name))
    solver = SolverPostProcessorPunct(solver)
    eval_ists_noali_headlines_train_batch(run_name, solver)


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    solve_w_post(args)
