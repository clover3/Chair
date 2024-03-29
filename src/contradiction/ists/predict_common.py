from typing import List

from alignment.base_ds import TextPairProblem
from contradiction.ists.save_path_helper import get_save_path
from contradiction.medical_claims.token_tagging.batch_solver_common import BatchTokenScoringSolverIF
from contradiction.medical_claims.token_tagging.online_solver_common import TokenScoringSolverIF
from contradiction.token_tagging.apply_solver_to_problems import make_ranked_list_w_solver, \
    make_ranked_list_w_batch_solver
from dataset_specific.ists.parse import iSTSProblem
from dataset_specific.ists.path_helper import load_ists_problems
from trainer_v2.chair_logging import c_log


def eval_ists_noali_headlines_train(run_name, solver: TokenScoringSolverIF):
    tag_type = "noali"
    save_path = get_save_path(run_name)
    ists_problems: List[iSTSProblem] = load_ists_problems("headlines", "train")
    problems: List[TextPairProblem] = [iSTSProblem.to_text_pair_problem(p) for p in ists_problems]
    make_ranked_list_w_solver(problems, run_name, save_path, tag_type, solver)


def eval_ists_noali_headlines_train_batch(run_name, solver: BatchTokenScoringSolverIF, mini=False):
    tag_type = "noali"
    ists_problems: List[iSTSProblem] = load_ists_problems("headlines", "train")
    if mini:
        run_name = run_name + "_mini"
        n_use = 10
        ists_problems = ists_problems[:n_use]
        c_log.info("Using first {} for prediction (mini)".format(n_use))

    save_path = get_save_path(run_name)
    problems: List[TextPairProblem] = [iSTSProblem.to_text_pair_problem(p) for p in ists_problems]
    make_ranked_list_w_batch_solver(problems, run_name, save_path, tag_type, solver)

