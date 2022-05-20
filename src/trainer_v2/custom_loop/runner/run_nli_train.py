import logging
import sys

from cpath import get_bert_config_path
from taskman_client.wrapper3 import report_run3
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.RunConfig2 import get_run_config2_nli_train, RunConfig2
from trainer_v2.custom_loop.dataset_factories import get_classification_dataset
from trainer_v2.custom_loop.modeling_common.bert_common import load_bert_config, ModelConfig
from trainer_v2.custom_loop.per_task.classification_runner import get_classification_runner
from trainer_v2.custom_loop.runner_if import RunnerIF
from trainer_v2.custom_loop.train_sub import tf_train_run
from trainer_v2.train_util.arg_flags import flags_parser


@report_run3
def main(args):
    c_log.info("Start train Classification")
    c_log.setLevel(logging.DEBUG)
    run_config: RunConfig2 = get_run_config2_nli_train(args)
    run_config.is_debug_run = True
    run_config.print_info()

    bert_params = load_bert_config(get_bert_config_path())
    model_config = ModelConfig()
    runner: RunnerIF = get_classification_runner(bert_params, model_config, run_config)

    def build_dataset(input_files):
        return get_classification_dataset(input_files, run_config, model_config)

    tf_train_run(run_config, runner, build_dataset)


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)


