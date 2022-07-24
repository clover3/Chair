from alignment.data_structure.matrix_scorer_if import MatrixScorerIF2From1
from alignment.ists_eval.eval_helper import solve_and_save_eval_ht
from alignment.ists_eval.eval_utils import score_matrix_to_alignment_by_threshold
from alignment.matrix_scorers.methods.all_nothing_scorer import AllOneScorer
from alignment.matrix_scorers2.methods.all_nothing_scorer import AllOneScorer2, AllZeroScorer2


def get_all_one_scorer():
    return MatrixScorerIF2From1(AllOneScorer())


def main():
    solver = AllOneScorer2()

    def score_matrix_to_alignment(matrix):
        return score_matrix_to_alignment_by_threshold(matrix, 0.5)
    solve_and_save_eval_ht(solver, "all_one", score_matrix_to_alignment)


def main2():
    solver = AllZeroScorer2()

    def score_matrix_to_alignment(matrix):
        return score_matrix_to_alignment_by_threshold(matrix, 0.5)
    solve_and_save_eval_ht(solver, "all_zero", score_matrix_to_alignment)


def main3():
    solver = get_all_one_scorer()

    def score_matrix_to_alignment(matrix):
        return score_matrix_to_alignment_by_threshold(matrix, 0.5)
    solve_and_save_eval_ht(solver, "all_one3", score_matrix_to_alignment)


if __name__ == "__main__":
    main3()