import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from contradiction.medical_claims.token_tagging.v2_solver_helper import solve_mismatch_ecc
from contradiction.medical_claims.token_tagging.solvers.nlits_batch_solver import get_batch_solver_nlits4
from taskman_client.wrapper3 import report_run3
from trainer_v2.train_util.arg_flags import flags_parser


@report_run3
def main(args):
    def solver_factory(run_config, target_label):
        solver = get_batch_solver_nlits4(run_config, "concat", target_label)
        return solver
    solve_mismatch_ecc(args, solver_factory)


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)
