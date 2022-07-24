from alignment.data_structure.matrix_scorer_if import MatrixScorerIF2From1
from alignment.ists_eval.eval_helper import solve_and_save_eval_ht
from alignment.ists_eval.eval_utils import score_matrix_to_alignment_by_threshold
from alignment.matrix_scorers.methods.get_word2vec_scorer import get_word2vec_scorer_from_d


def get_solver():
    return MatrixScorerIF2From1(get_word2vec_scorer_from_d())


def main():
    solver = get_solver()

    def score_matrix_to_alignment(matrix):
        return score_matrix_to_alignment_by_threshold(matrix, 0.5)
    solve_and_save_eval_ht(solver, "word2vec", score_matrix_to_alignment)


if __name__ == "__main__":
    main()