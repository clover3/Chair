from typing import List, Iterable, Callable, Dict, Tuple, Set

from contradiction.medical_claims.token_tagging.online_solver_common import make_ranked_list_w_solver2
from contradiction.medical_claims.token_tagging.path_helper import get_save_path2
from contradiction.medical_claims.token_tagging.problem_loader import load_alamri_problem, AlamriProblem
from contradiction.medical_claims.token_tagging.solvers.search_solver import WordSegSolver
from trainer_v2.keras_server.name_short_cuts import get_nli14_client


def word_seg_solver(run_name, tag_type, target_idx):
    save_path = get_save_path2(run_name, tag_type)
    problems: List[AlamriProblem] = load_alamri_problem()
    predict_fn = get_nli14_client().request_multiple
    solver = WordSegSolver(target_idx, predict_fn)
    make_ranked_list_w_solver2(problems, run_name, save_path, tag_type, solver)


def main():
    tag_type = "mismatch"
    run_name = "word_seg"
    word_seg_solver(run_name, tag_type, 1)


def main():
    tag_type = "conflict"
    run_name = "word_seg"
    word_seg_solver(run_name, tag_type, 2)


if __name__ == "__main__":
    main()