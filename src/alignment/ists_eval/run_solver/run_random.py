from alignment.ists_eval.matrix_eval_helper import solve_and_save_eval_ht
from alignment.ists_eval.prediction_helper import score_matrix_to_alignment_by_threshold
from alignment.matrix_scorers2.methods.random_scorer import RandomOne


def main():
    solver = RandomOne()

    def score_matrix_to_alignment(matrix):
        return score_matrix_to_alignment_by_threshold(matrix, 0.5)
    solve_and_save_eval_ht(solver, "random_one", score_matrix_to_alignment)


if __name__ == "__main__":
    main()