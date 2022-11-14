from alignment.ists_eval.chunked_solver.random_solver import RandomSolver, LocationSolver
from alignment.ists_eval.chunked_eval import chunked_solve_and_save_eval


def main():
    solver = LocationSolver()
    chunked_solve_and_save_eval(solver, "location_chunked", "headlines", "train")
    solver = RandomSolver()
    chunked_solve_and_save_eval(solver, "random_chunked", "headlines", "train")



if __name__ == "__main__":
    main()