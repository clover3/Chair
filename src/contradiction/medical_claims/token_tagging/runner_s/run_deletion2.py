from typing import List

from bert_api.task_clients.nli_interface.nli_predictors import get_nli_client
from contradiction.medical_claims.token_tagging.online_solver_common import make_ranked_list_w_solver2
from contradiction.medical_claims.token_tagging.path_helper import get_save_path2
from contradiction.medical_claims.token_tagging.problem_loader import AlamriProblem, load_alamri_problem
from contradiction.medical_claims.token_tagging.solvers.deletion_solver import DeletionSolver, DeletionSolverKeras
from data_generator.NLI.enlidef import ENTAILMENT, CONTRADICTION, NEUTRAL
from trainer_v2.keras_server.name_short_cuts import get_nli14_direct
from trainer_v2.train_util.get_tpu_strategy import get_strategy


def do_for(tag_type, target_idx):
    problems: List[AlamriProblem] = load_alamri_problem()
    run_name = "deletion"
    save_path = get_save_path2(run_name, tag_type)
    predict_fn = get_nli14_direct(get_strategy())
    solver = DeletionSolverKeras(predict_fn, target_idx)
    make_ranked_list_w_solver2(problems, run_name, save_path, tag_type, solver)


def main():
    target_idx = NEUTRAL
    tag_type = "mismatch"
    do_for(tag_type, target_idx)
    #
    # target_idx = CONTRADICTION
    # tag_type = "conflict"
    # do_for(tag_type, target_idx)


if __name__ == "__main__":
    main()
