import os
import time
from typing import List

from contradiction.medical_claims.token_tagging.online_solver_common import make_ranked_list_w_solver2
from contradiction.medical_claims.token_tagging.path_helper import get_save_path2
from contradiction.medical_claims.token_tagging.problem_loader import load_alamri_problem, AlamriProblem
from contradiction.medical_claims.token_tagging.solvers.nlits_solver import get_nli_ts34_solver, LocalDecisionNLICore, \
    NLITSSolver, CachingAdapter
from cpath import common_model_dir_root
from data_generator.NLI.enlidef import NEUTRAL, CONTRADICTION
from misc_lib import tprint
from taskman_client.task_proxy import get_local_machine_name
from trainer_v2.custom_loop.neural_network_def.siamese import ModelConfig200_200
from trainer_v2.train_util.get_tpu_strategy import get_strategy


def main():
    tag_type = "mismatch"
    target_label = NEUTRAL
    run_name = "nlits"
    save_path = get_save_path2(run_name, tag_type)
    problems: List[AlamriProblem] = load_alamri_problem()
    print("Using {} problems".format(len(problems)))
    machine_name = get_local_machine_name()
    is_tpu = machine_name not in ["GOSFORD", "ingham.cs.umass.edu"]

    if is_tpu:
        model_path = "gs://clovertpu/training/model/nli_ts_run34_0/model_25000"
        model_path = '/home/youngwookim/code/Chair/nli_ts_run34_0/model_25000'
        strategy = get_strategy(True, "local")
    else:
        model_path = os.path.join(common_model_dir_root, 'runs', "nli_ts_run34_0", "model_25000")
        strategy = get_strategy(False, "")
    with strategy.scope():
        model_config = ModelConfig200_200()
        ld_core = LocalDecisionNLICore(model_path, strategy)

    adapter = CachingAdapter(ld_core.predict, [2, 3])
    dummy_solver = NLITSSolver(adapter.register_payloads,
                               model_config.max_seq_length1,
                               model_config.max_seq_length2, target_label)

    tprint("Running dummy solver..")
    make_ranked_list_w_solver2(problems, run_name + '_dummy', save_path, tag_type, dummy_solver)
    tprint("Now running batch predict.")
    st = time.time()
    adapter.batch_predict()
    print("maybe Tf time", time.time() - st)
    real_solver = NLITSSolver(adapter.predict,
                              model_config.max_seq_length1,
                              model_config.max_seq_length2, target_label)

    make_ranked_list_w_solver2(problems, run_name, save_path, tag_type, real_solver)

    # print("Total time:", solver.elapsed_all)
    # print("TF time:", solver.elapsed_tf)


def solve_conflict():
    tag_type = "conflict"
    run_name = "nlits"
    save_path = get_save_path2(run_name, tag_type)
    problems: List[AlamriProblem] = load_alamri_problem()
    model_path = os.path.join(common_model_dir_root, 'runs', "nli_ts_run34_0", "model_25000")
    solver = get_nli_ts34_solver(model_path, CONTRADICTION)
    make_ranked_list_w_solver2(problems, run_name, save_path, tag_type, solver)


if __name__ == "__main__":
    main()
