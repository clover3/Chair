from typing import List

from contradiction.medical_claims.token_tagging.batch_solver_common import make_ranked_list_w_batch_solver
from contradiction.medical_claims.token_tagging.eval_helper import do_ecc_eval_w_trec_eval
from contradiction.medical_claims.token_tagging.path_helper import get_save_path2
from contradiction.medical_claims.token_tagging.problem_loader import AlamriProblem, load_alamri_problem
from data_generator.NLI.enlidef import NEUTRAL
from trainer_v2.custom_loop.run_config2 import get_run_config2


def solve_mismatch_ecc(args, solver_factory):
    tag_type = "mismatch"
    target_label = NEUTRAL
    run_config = get_run_config2(args)
    run_config.print_info()
    solver = solver_factory(run_config, target_label)
    run_name = args.run_name
    save_path = get_save_path2(run_name, tag_type)
    problems: List[AlamriProblem] = load_alamri_problem()
    print("Using {} problems".format(len(problems)))
    make_ranked_list_w_batch_solver(problems, run_name, save_path, tag_type, solver)
    do_ecc_eval_w_trec_eval(run_name, tag_type)