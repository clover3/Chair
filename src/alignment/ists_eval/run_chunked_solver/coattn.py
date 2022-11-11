from alignment.ists_eval.chunked_eval import chunked_solve_and_save_eval
from alignment.ists_eval.chunked_solver.coattention import get_co_attention_chunked_solver


def main():
    solver = get_co_attention_chunked_solver()
    chunked_solve_and_save_eval(solver, "coattention_chunked", "headlines", "train")


if __name__ == "__main__":
    main()
