from contradiction.esnli.load_esnli import load_esnli
from contradiction.esnli.path_helper import get_save_path_ex
from typing import List

from contradiction.medical_claims.token_tagging.batch_solver_common import BatchTokenScoringSolverIF
from contradiction.medical_claims.token_tagging.online_solver_common import TokenScoringSolverIF
from contradiction.mnli_ex.nli_ex_common import make_ranked_list_w_solver_nli, make_ranked_list_w_batch_solver, \
    NLIExEntry



def solve_esnli_tag(split, run_name, solver: TokenScoringSolverIF, tag_type):
    problems: List[NLIExEntry] = load_esnli(split, tag_type)
    save_path = get_save_path_ex(split, run_name, tag_type)
    make_ranked_list_w_solver_nli(problems, solver, run_name, save_path)


def solve_esnli_token_batch(split, run_name, solver: BatchTokenScoringSolverIF, tag_type):
    problems: List[NLIExEntry] = load_esnli(split, tag_type)
    save_path = get_save_path_ex(split, run_name, tag_type)
    make_ranked_list_w_batch_solver(problems, run_name, save_path, solver)
