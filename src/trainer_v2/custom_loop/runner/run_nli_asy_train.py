import logging
import sys

from cpath import get_bert_config_path
from taskman_client.wrapper3 import report_run3
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.RunConfig2 import get_run_config2_nli_train, RunConfig2
from trainer_v2.custom_loop.dataset_factories import get_two_seg_data
from trainer_v2.custom_loop.modeling_common.assymetric import ModelConfig2Seg
from trainer_v2.custom_loop.modeling_common.bert_common import load_bert_config
from trainer_v2.custom_loop.per_task.classification_runner_factory import get_classification_asym_runner
from trainer_v2.custom_loop.train_sub import tf_train_run
from trainer_v2.train_util.arg_flags import flags_parser


@report_run3
def main(args):
    c_log.info("Start train Classification asymmetric")
    c_log.setLevel(logging.DEBUG)
    run_config: RunConfig2 = get_run_config2_nli_train(args)
    run_config.is_debug_run = True
    run_config.print_info()

    bert_params = load_bert_config(get_bert_config_path())
    model_config = ModelConfig2Seg()
    runner = get_classification_asym_runner(bert_params, model_config, run_config)

    def dataset_factory(input_files):
        return get_two_seg_data(input_files, run_config, model_config)

    tf_train_run(run_config, runner, dataset_factory)


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)


