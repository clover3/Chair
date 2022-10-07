import os
import sys

from trainer_v2.chair_logging import c_log

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from contradiction.medical_claims.token_tagging.solvers.nlits_batch_solver import get_batch_solver_nlits6

from typing import List

from contradiction.medical_claims.token_tagging.batch_solver_common import make_ranked_list_w_batch_solver
from contradiction.medical_claims.token_tagging.eval_helper import do_ecc_eval_w_trec_eval
from contradiction.medical_claims.token_tagging.path_helper import get_save_path2
from contradiction.medical_claims.token_tagging.problem_loader import AlamriProblem, load_alamri_problem
from data_generator.NLI.enlidef import NEUTRAL
from trainer_v2.custom_loop.run_config2 import get_run_config2

from contradiction.medical_claims.token_tagging.v2_solver_helper import solve_mismatch_ecc
from taskman_client.wrapper3 import report_run3
from trainer_v2.train_util.arg_flags import flags_parser



def solve_mismatch_ecc(args, solver_factory):
    tag_type = "mismatch"
    target_label = NEUTRAL
    run_config = get_run_config2(args)
    run_config.print_info()
    solver = solver_factory(run_config, target_label)
    run_name = args.run_name
    save_path = get_save_path2(run_name, tag_type)
    problems: List[AlamriProblem] = load_alamri_problem()
    problems = problems[:1]
    payload = []
    c_log.info("note 2")
    for p in problems:
        input_per_problem = p.text1.split(), p.text2.split()
        payload.append(input_per_problem)
    batch_output = solver.solve(payload)

    # do_ecc_eval_w_trec_eval(run_name, tag_type)


@report_run3
def main(args):
    c_log.info("Start {}".format(__file__))
    def solver_factory(run_config, target_label):
        solver = get_batch_solver_nlits6(run_config, "concat", target_label)
        return solver
    solve_mismatch_ecc(args, solver_factory)



if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)
