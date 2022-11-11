import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from trainer_v2.custom_loop.run_config2 import get_eval_run_config2


from alignment.data_structure.matrix_scorer_if import MatrixScorerIF2
from alignment.ists_eval.matrix_eval_helper import solve_and_save_eval_headlines_2d
from alignment.matrix_scorers2.methods.nlits_scorer import get_nlits_2dsolver
from trainer_v2.train_util.arg_flags import flags_parser


def main(args):
    run_config = get_eval_run_config2(args)
    run_config.print_info()
    solver: MatrixScorerIF2 = get_nlits_2dsolver(run_config, "concat")
    run_name = "nlits"
    solve_and_save_eval_headlines_2d(solver, run_name)


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)

