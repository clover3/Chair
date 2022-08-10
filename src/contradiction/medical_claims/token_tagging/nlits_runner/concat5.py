import os
import sys


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from trainer_v2.chair_logging import c_log
from contradiction.medical_claims.token_tagging.v2_solver_helper import solve_mismatch_ecc
from contradiction.medical_claims.token_tagging.solvers.nlits_batch_solver import get_batch_solver_nlits5
from taskman_client.wrapper3 import report_run3
from trainer_v2.train_util.arg_flags import flags_parser


@report_run3
def main(args):
    c_log.info("Start {}".format(__file__))
    def solver_factory(run_config, target_label):
        solver = get_batch_solver_nlits5(run_config, "concat", target_label)
        return solver
    solve_mismatch_ecc(args, solver_factory)


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)
