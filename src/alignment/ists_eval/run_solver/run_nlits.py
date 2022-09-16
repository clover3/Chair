import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from trainer_v2.custom_loop.run_config2 import get_run_config2


from alignment.data_structure.matrix_scorer_if import MatrixScorerIF2
from alignment.ists_eval.eval_helper import solve_and_save_eval_mini, solve_and_save_eval_ht, solve_and_save_eval_mini50
from alignment.ists_eval.eval_utils import score_matrix_to_alignment_by_threshold
from alignment.matrix_scorers2.methods.nlits_scorer import get_nlits_2dsolver
from taskman_client.wrapper3 import report_run3
from trainer_v2.train_util.arg_flags import flags_parser

@report_run3
def main(args):
    run_config = get_run_config2(args)
    run_config.print_info()
    solver: MatrixScorerIF2 = get_nlits_2dsolver(run_config, "concat")

    def score_matrix_to_alignment(matrix):
        return score_matrix_to_alignment_by_threshold(matrix, 0.05)
    # solve_and_save_eval_ht(solver, "nlits", score_matrix_to_alignment)
    # solve_and_save_eval_mini(solver, "nlits_mini", score_matrix_to_alignment)
    solve_and_save_eval_mini50(solver, "nlits_mini50", score_matrix_to_alignment)


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)

