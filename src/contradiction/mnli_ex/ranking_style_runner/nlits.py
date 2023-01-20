import sys
from contradiction.medical_claims.token_tagging.solvers.nlits_batch_solver import get_batch_solver_nlits6
from contradiction.mnli_ex.ranking_style_helper import solve_mnli_tag_batch
from data_generator.NLI.enlidef import get_target_class, enli_tags
from taskman_client.wrapper3 import report_run3
from trainer_v2.custom_loop.run_config2 import get_run_config_for_predict
from trainer_v2.train_util.arg_flags import flags_parser


def do_for_label(run_config, tag_type, split):
    target_idx = get_target_class(tag_type)
    solver = get_batch_solver_nlits6(run_config, "concat", target_idx)
    run_name = "nlits87"
    solve_mnli_tag_batch(split, run_name, solver, tag_type)


@report_run3
def main(args):
    run_config = get_run_config_for_predict(args)
    for split in ["dev", "test"]:
        for tag_type in enli_tags:
            do_for_label(run_config, tag_type, split)


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)
