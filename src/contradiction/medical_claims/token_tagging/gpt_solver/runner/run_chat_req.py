import os

from contradiction.medical_claims.token_tagging.gpt_solver.get_chat_gpt_solver import get_chat_gpt_requester
from contradiction.medical_claims.token_tagging.problem_loader import AlamriProblem, load_alamri_problem
from contradiction.medical_claims.token_tagging.runner_s.run_exact_match import solve_and_save
from typing import List, Iterable, Callable, Dict, Tuple, Set

from misc_lib import TEL
from utils.open_ai_api import ENGINE_GPT_3_5


def apply_solver(problems: List[AlamriProblem],
                 solver):
    def problem_to_scores(p: AlamriProblem):
        return solver.solve_from_text(p.text1, p.text2)

    for p in TEL(problems):
        scores1, scores2 = problem_to_scores(p)


def main():
    tag_type = "mismatch"
    engine = ENGINE_GPT_3_5
    run_name = engine
    print("Get requester")
    solver = get_chat_gpt_requester(engine, tag_type)
    problems: List[AlamriProblem] = load_alamri_problem()[:1]
    print("apply solver")
    apply_solver(problems, solver)


if __name__ == "__main__":
    main()
