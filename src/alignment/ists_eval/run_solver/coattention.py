from alignment.ists_eval.matrix_eval_helper import solve_and_save_eval_headlines_2d
from alignment.matrix_scorers2.methods.coattention_solver import get_co_attention_solver


def main():
    run_name = "coattn"
    solver = get_co_attention_solver()
    solve_and_save_eval_headlines_2d(solver, run_name)


if __name__ == "__main__":
    main()
