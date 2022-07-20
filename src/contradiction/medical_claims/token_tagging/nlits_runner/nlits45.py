import os
import sys
from typing import List

from contradiction.medical_claims.token_tagging.solvers.nlits_batch_solver import get_batch_solver_nlits
from taskman_client.wrapper3 import report_run3
from trainer_v2.custom_loop.run_config2 import get_run_config2, RunConfig2
from trainer_v2.train_util.arg_flags import flags_parser

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from contradiction.medical_claims.token_tagging.batch_solver_common import make_ranked_list_w_batch_solver
from contradiction.medical_claims.token_tagging.path_helper import get_save_path2
from contradiction.medical_claims.token_tagging.problem_loader import load_alamri_problem, AlamriProblem
from data_generator.NLI.enlidef import NEUTRAL


def run_nlits_solver(run_config: RunConfig2, tag_type, target_label: int):
    solver = get_batch_solver_nlits(run_config, "two_seg", target_label)
    run_name = run_config.common_run_config.run_name
    save_path = get_save_path2(run_name, tag_type)
    problems: List[AlamriProblem] = load_alamri_problem()
    print("Using {} problems".format(len(problems)))
    make_ranked_list_w_batch_solver(problems, run_name, save_path, tag_type, solver)


@report_run3
def main(args):
    tag_type = "mismatch"
    target_label = NEUTRAL
    run_config = get_run_config2(args)
    run_nlits_solver(run_config, tag_type, target_label)


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)
