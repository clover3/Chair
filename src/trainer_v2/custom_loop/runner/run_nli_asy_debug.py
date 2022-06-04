import logging
import sys

from cpath import get_bert_config_path
from taskman_client.wrapper3 import report_run3
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.dataset_factories import build_dataset_repeat_segs
from trainer_v2.custom_loop.modeling_common.bert_common import load_bert_config, ModelConfig
from trainer_v2.custom_loop.neural_network_def.inner_network import AsymDebug
from trainer_v2.custom_loop.per_task.trainer import Trainer
from trainer_v2.custom_loop.run_config2 import get_run_config2_nli, RunConfig2
from trainer_v2.custom_loop.train_loop import tf_run_train
from trainer_v2.train_util.arg_flags import flags_parser


@report_run3
def main(args):
    c_log.info("Start train Classification asymmetric debug")
    c_log.setLevel(logging.DEBUG)
    run_config: RunConfig2 = get_run_config2_nli(args)
    run_config.is_debug_run = True
    run_config.print_info()

    bert_params = load_bert_config(get_bert_config_path())
    model_config = ModelConfig()
    inner = AsymDebug()
    trainer = Trainer(bert_params, model_config, run_config, inner)

    def dataset_factory(input_files):
        return build_dataset_repeat_segs(input_files, run_config, model_config)

    tf_run_train(run_config, trainer, dataset_factory)


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)


