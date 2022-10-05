import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from contradiction.ists.predict_common import eval_ists_noali_headlines_train_batch
from contradiction.medical_claims.token_tagging.solvers.nlits_batch_solver import SolverPostProcessorPunct, get_batch_solver_nlits7
from taskman_client.wrapper3 import report_run3
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.run_config2 import get_run_config2
from trainer_v2.train_util.arg_flags import flags_parser


@report_run3
def solve_w_post(args):
    c_log.info("Start {}".format(__file__))
    run_config = get_run_config2(args)
    solver = get_batch_solver_nlits7(run_config, "concat")
    run_name = "nlits_punc"
    c_log.info("run name= {}".format(run_name))
    solver = SolverPostProcessorPunct(solver)
    eval_ists_noali_headlines_train_batch(run_name, solver)


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    solve_w_post(args)

