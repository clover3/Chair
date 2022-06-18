from typing import List

from krovetzstemmer import Stemmer

from arg.clueweb12_B13_termstat import load_clueweb12_B13_termstat_stemmed, cdf
from contradiction.medical_claims.token_tagging.online_solver_common import make_ranked_list_w_solver2
from contradiction.medical_claims.token_tagging.path_helper import get_save_path2
from contradiction.medical_claims.token_tagging.problem_loader import load_alamri_problem, AlamriProblem
from contradiction.medical_claims.token_tagging.solvers.exact_match_solver import ExactMatchSolver, \
    ExactMatchSTHandleSolver
from contradiction.medical_claims.token_tagging.solvers.idf_solver import TF_IDF


def main_old():
    tag_type = "mismatch"
    tag_type = "conflict"
    run_name = "exact_match"
    solver = ExactMatchSolver()
    save_path = get_save_path2(run_name, tag_type)
    problems: List[AlamriProblem] = load_alamri_problem()
    make_ranked_list_w_solver2(problems, run_name, save_path, tag_type, solver)


def main():
    tag_type = "mismatch"
    run_name = "exact_match_st_handle"
    solver = ExactMatchSTHandleSolver()
    solve_and_save(run_name, solver, tag_type)


def solve_and_save(run_name, solver, tag_type):
    save_path = get_save_path2(run_name, tag_type)
    problems: List[AlamriProblem] = load_alamri_problem()
    make_ranked_list_w_solver2(problems, run_name, save_path, tag_type, solver)


def run_tf_idf():
    solver = get_tf_idf_solver()
    tag_type = "mismatch"
    run_name = "tf_idf"
    solve_and_save(run_name, solver, tag_type)


def get_tf_idf_solver():
    tf, df = load_clueweb12_B13_termstat_stemmed()
    stemmer = Stemmer()
    solver = TF_IDF(df, cdf, stemmer)
    return solver


if __name__ == "__main__":
    run_tf_idf()
