from contradiction.medical_claims.token_tagging.gpt_solver.get_instruct_gpt_solver import get_gpt_file_solver_conflict
from contradiction.medical_claims.token_tagging.util import solve_and_save


def main():
    tag_type = "conflict"
    run_name = "davinci"
    solver = get_gpt_file_solver_conflict()
    solve_and_save(run_name, solver, tag_type)


if __name__ == "__main__":
    main()
