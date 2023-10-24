import sys

from taskman_client.wrapper3 import report_run3
from trainer_v2.custom_loop.definitions import ModelConfig512_1
from trainer_v2.custom_loop.per_task.local_decision_server import run_local_decision_server
from trainer_v2.custom_loop.run_config2 import get_eval_run_config2
from trainer_v2.custom_loop.train_loop_helper import get_strategy_from_config
from trainer_v2.train_util.arg_flags import flags_parser


@report_run3
def run_server(args):
    model_config = ModelConfig512_1()
    run_config = get_eval_run_config2(args)
    strategy = get_strategy_from_config(run_config)
    run_local_decision_server(run_config, model_config, strategy)


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    run_server(args)
