from bert_api.task_clients.nli_interface.nli_predictors import get_nli_client
from contradiction.medical_claims.token_tagging.solvers.deletion_solver import DeletionSolver
from contradiction.mnli_ex.ranking_style_helper import solve_mnli_tag
from data_generator.NLI.enlidef import ENTAILMENT, CONTRADICTION


def do_for(tag_type, target_idx):
    run_name = "deletion"
    predict_fn = get_nli_client("localhost")
    solver = DeletionSolver(predict_fn, target_idx)
    split = "test"
    solve_mnli_tag(split, run_name, solver, tag_type)


def main():
    target_idx = ENTAILMENT
    tag_type = "match"
    do_for(tag_type, target_idx)
    target_idx = CONTRADICTION
    tag_type = "conflict"
    do_for(tag_type, target_idx)


if __name__ == "__main__":
    main()
