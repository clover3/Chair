from alignment.data_structure.matrix_scorer_if import MatrixScorerIF2
from alignment.ists_eval.eval_helper import solve_and_save_eval_mini
from alignment.ists_eval.eval_utils import score_matrix_to_alignment_by_threshold
from alignment.matrix_scorers2.methods.probe_scorer import get_probe_solver


def main():
    solver: MatrixScorerIF2 = get_probe_solver()

    def score_matrix_to_alignment(matrix):
        return score_matrix_to_alignment_by_threshold(matrix, 0)
    # solve_and_save_eval_ht(solver, "probe", score_matrix_to_alignment)
    solve_and_save_eval_mini(solver, "probe_mini", score_matrix_to_alignment)


if __name__ == "__main__":
    main()
