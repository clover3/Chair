from contradiction.medical_claims.token_tagging.runner_s.run_exact_match import get_tf_idf_solver
from contradiction.mnli_ex.ranking_style_helper import solve_mnli_tag


def main():
    solver = get_tf_idf_solver()
    run_name = "tf_idf"
    tag_type = "mismatch"
    split = "test"
    solve_mnli_tag(split, run_name, solver, tag_type)


if __name__ == "__main__":
    main()
