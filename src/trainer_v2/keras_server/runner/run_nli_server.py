import sys
from port_info import KERAS_NLI_PORT

from trainer_v2.custom_loop.definitions import ModelConfig300_3
from trainer_v2.custom_loop.run_config2 import get_eval_run_config2
from trainer_v2.custom_loop.train_loop_helper import get_strategy_from_config
from trainer_v2.keras_server.bert_like_server import run_keras_bert_like_server
from trainer_v2.train_util.arg_flags import flags_parser


def run_server(args):
    run_config = get_eval_run_config2(args)
    strategy = get_strategy_from_config(run_config)
    model_path = run_config.get_model_path()
    model_config = ModelConfig300_3()
    port = KERAS_NLI_PORT
    run_keras_bert_like_server(port, model_path, model_config, strategy)


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    run_server(args)
