from alignment.ists_eval.chunked_eval import chunked_solve_and_save_eval
from alignment.ists_eval.chunked_solver.nli_v3 import get_solver
from dataset_specific.ists.split_info import ists_enum_split_genre_combs


def main():
    nli_type = "em"
    label_predictor_type = "wo_context"
    run_name = f"{nli_type}_{label_predictor_type}"
    solver = get_solver(nli_type, label_predictor_type)
    for split, genre in ists_enum_split_genre_combs():
        chunked_solve_and_save_eval(solver, run_name, genre, split)


if __name__ == "__main__":
    main()