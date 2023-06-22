from typing import List, Iterable, Callable, Dict, Tuple, Set

from contradiction.medical_claims.token_tagging.gpt_solver.get_chat_gpt_solver import get_chat_gpt_file_solver
from contradiction.medical_claims.token_tagging.online_solver_common import make_ranked_list_w_solver2
from contradiction.medical_claims.token_tagging.path_helper import get_save_path2
from contradiction.medical_claims.token_tagging.util import solve_and_save, apply_solver
from utils.open_ai_api import ENGINE_GPT_3_5
from contradiction.medical_claims.token_tagging.gpt_solver.get_chat_gpt_solver import get_chat_gpt_requester
from contradiction.medical_claims.token_tagging.problem_loader import AlamriProblem, load_alamri_problem


def main():
    tag_type = "conflict"
    engine = ENGINE_GPT_3_5
    run_name = engine
    solver = get_chat_gpt_file_solver(engine, tag_type)
    problems: List[AlamriProblem] = load_alamri_problem()
    print("apply solver")
    save_path = get_save_path2(run_name, tag_type)
    make_ranked_list_w_solver2(problems, run_name, save_path, tag_type, solver)


if __name__ == "__main__":
    main()
