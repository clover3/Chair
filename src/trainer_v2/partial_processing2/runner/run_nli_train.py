import logging
import sys

from cpath import get_bert_config_path
from taskman_client.wrapper3 import report_run3
from trainer_v2.chair_logging import c_log
from trainer_v2.partial_processing2.RunConfig2 import get_run_config2_nli_train, RunConfig2
from trainer_v2.partial_processing2.modeling_common.bert_common import load_bert_config, ModelConfig
from trainer_v2.partial_processing2.nli_runner import NLIRunner
from trainer_v2.partial_processing2.train_sub import tf_run
from trainer_v2.train_util.arg_flags import flags_parser


@report_run3
def main(args):
    c_log.info("Start train NLI")
    c_log.setLevel(logging.DEBUG)
    run_config: RunConfig2 = get_run_config2_nli_train(args)
    run_config.is_debug_run = True
    run_config.print_info()

    bert_params = load_bert_config(get_bert_config_path())
    model_config = ModelConfig()
    runner = NLIRunner(bert_params, model_config, run_config)
    tf_run(run_config, runner)


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)


