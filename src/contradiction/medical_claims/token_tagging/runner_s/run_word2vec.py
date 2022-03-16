from typing import List

from contradiction.medical_claims.token_tagging.online_solver_common import make_ranked_list_w_solver2
from contradiction.medical_claims.token_tagging.path_helper import get_save_path2
from contradiction.medical_claims.token_tagging.problem_loader import load_alamri_problem, AlamriProblem
from contradiction.medical_claims.token_tagging.solvers.word2vec_solver import get_word2vec_solver, \
    get_pubmed_word2vec_solver, get_word2vec_em_solver, get_w2v_antonym
from misc_lib import tprint


def main():
    tag_type = "mismatch"
    tag_type = "conflict"
    run_name = "word2vec"
    save_path = get_save_path2(run_name, tag_type)
    problems: List[AlamriProblem] = load_alamri_problem()
    tprint("Building solver")
    solver = get_word2vec_solver()
    tprint("Building solver DONE")
    make_ranked_list_w_solver2(problems, run_name, save_path, tag_type, solver)


def main3():
    tag_type = "mismatch"
    tag_type = "conflict"
    run_name = "word2vec_em"
    save_path = get_save_path2(run_name, tag_type)
    problems: List[AlamriProblem] = load_alamri_problem()
    tprint("Building solver")
    solver = get_word2vec_em_solver()
    tprint("Building solver DONE")
    make_ranked_list_w_solver2(problems, run_name, save_path, tag_type, solver)


def main2():
    tag_type = "mismatch"
    run_name = "word2vec_pm"
    save_path = get_save_path2(run_name, tag_type)
    problems: List[AlamriProblem] = load_alamri_problem()
    tprint("Building solver")
    solver = get_pubmed_word2vec_solver()
    tprint("Building solver DONE")
    make_ranked_list_w_solver2(problems, run_name, save_path, tag_type, solver)


def main4():
    tag_type = "conflict"
    run_name = "word2vec_antonym"
    save_path = get_save_path2(run_name, tag_type)
    problems: List[AlamriProblem] = load_alamri_problem()
    tprint("Building solver")
    solver = get_w2v_antonym()
    tprint("Building solver DONE")
    make_ranked_list_w_solver2(problems, run_name, save_path, tag_type, solver)


if __name__ == "__main__":
    main4()
