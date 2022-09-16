import sys

from contradiction.medical_claims.token_tagging.solvers.senli_solver import get_senli_solver, get_senli_solver_aym
from contradiction.mnli_ex.ranking_style_helper import solve_mnli_tag


def main():
    split = sys.argv[1]
    tag_type = sys.argv[2]
    run_name = "senli"
    solver = get_senli_solver_aym(tag_type)
    solve_mnli_tag(split, run_name, solver, tag_type)


if __name__ == "__main__":
    main()
