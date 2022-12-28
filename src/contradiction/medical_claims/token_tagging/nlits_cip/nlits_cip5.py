import os
import sys

from contradiction.medical_claims.token_tagging.solvers.nlits_batch_solver2 import get_batch_solver_nlits_cip5
from data_generator.NLI.enlidef import NEUTRAL
from trainer_v2.chair_logging import c_log

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


from contradiction.medical_claims.token_tagging.v2_solver_helper import solve_ecc
from taskman_client.wrapper3 import report_run3
from trainer_v2.train_util.arg_flags import flags_parser


@report_run3
def main(args):
    c_log.info("Start {}".format(os.path.basename(__file__)))
    def solver_factory(run_config, target_label):
        solver = get_batch_solver_nlits_cip5(run_config, "concat", target_label)
        return solver
    solve_ecc(args, solver_factory, "mismatch", NEUTRAL)


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)
