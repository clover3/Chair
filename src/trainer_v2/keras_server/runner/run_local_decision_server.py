import sys

from trainer_v2.custom_loop.definitions import ModelConfig600_3
from trainer_v2.custom_loop.per_task.local_decision_server import run_local_decision_server
from trainer_v2.custom_loop.per_task.nli_ts_util import dataset_factory_600_3
from trainer_v2.custom_loop.run_config2 import get_eval_run_config2
from trainer_v2.custom_loop.train_loop_helper import get_strategy_from_config
from trainer_v2.train_util.arg_flags import flags_parser


def run_server(args):
    run_config = get_eval_run_config2(args)
    strategy = get_strategy_from_config(run_config)
    model_config = ModelConfig600_3()
    run_local_decision_server(run_config, model_config, strategy)



if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    run_server(args)
