import sys
from contradiction.medical_claims.token_tagging.solvers.nlits_batch_solver import get_batch_solver_nlits6
from contradiction.solve_run_helper import solve_esnli_token_batch
from data_generator.NLI.enlidef import get_target_class, enli_tags
from data_generator.job_runner import WorkerInterface
from epath import job_man_dir
from job_manager.job_runner_with_server import JobRunnerF
from taskman_client.wrapper3 import report_run3
from trainer_v2.custom_loop.run_config2 import get_run_config_for_predict
from trainer_v2.train_util.arg_flags import flags_parser


def do_for_label(run_config, tag_type, split):
    target_idx = get_target_class(tag_type)
    solver = get_batch_solver_nlits6(run_config, "concat", target_idx)
    run_name = "nlits87"
    solve_esnli_token_batch(split, run_name, solver, tag_type)


def get_todo():
    entries = []
    for split in ["dev", "test"]:
        for tag_type in enli_tags:
            entries.append((tag_type, split))
    return entries


@report_run3
def main(args):
    run_config = get_run_config_for_predict(args)
    todo_list = get_todo()

    def work_fn(job_id):
        tag_type, split = todo_list[job_id]
        do_for_label(run_config, tag_type, split)

    num_jobs = len(todo_list)
    runner = JobRunnerF(job_man_dir, num_jobs, "esnli_ts2", work_fn)
    runner.auto_runner()


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)
