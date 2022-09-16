from bert_api.task_clients.nli_interface.nli_predictors import get_nli_cache_client, get_nli_client
from contradiction.medical_claims.token_tagging.solvers.deletion_solver import DeletionSolver
from contradiction.mnli_ex.ranking_style_helper import solve_mnli_tag


def main():
    predict_fn = get_nli_client("localhost")
    run_name = "deletion"
    solver = DeletionSolver(predict_fn, 0)
    tag_type = "mismatch"
    split = "test"
    solve_mnli_tag(split, run_name, solver, tag_type)


if __name__ == "__main__":
    main()