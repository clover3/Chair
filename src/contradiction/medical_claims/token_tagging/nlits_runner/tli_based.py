import os
import sys

from contradiction.medical_claims.token_tagging.solvers.nlits_dual_batch import get_batch_solver_tli_based
from trainer_v2.chair_logging import c_log

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from contradiction.medical_claims.token_tagging.solvers.nlits_batch_solver import get_batch_solver_nlits6


from contradiction.medical_claims.token_tagging.v2_solver_helper import solve_mismatch_ecc
from taskman_client.wrapper3 import report_run3
from trainer_v2.train_util.arg_flags import flags_parser


@report_run3
def main(args):
    c_log.info("Start {}".format(__file__))
    def solver_factory(run_config, target_label):
        return get_batch_solver_tli_based(run_config, target_label)
    solve_mismatch_ecc(args, solver_factory)


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)
