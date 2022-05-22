import sys

from cpath import get_bert_config_path
from taskman_client.wrapper3 import report_run3
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.dataset_factories import get_two_seg_data
from trainer_v2.custom_loop.modeling_common.assymetric import ModelConfig2Seg
from trainer_v2.custom_loop.modeling_common.bert_common import load_bert_config
from trainer_v2.custom_loop.per_task.inner_network import Siamese
from trainer_v2.custom_loop.per_task.trainer import Trainer
from trainer_v2.custom_loop.run_config2 import RunConfig2, get_run_config2_nli
from trainer_v2.custom_loop.train_loop import tf_run
from trainer_v2.train_util.arg_flags import flags_parser


@report_run3
def main(args):
    c_log.info("Main classification-siamese")
    run_config: RunConfig2 = get_run_config2_nli(args)
    run_config.print_info()

    bert_params = load_bert_config(get_bert_config_path())
    model_config = ModelConfig2Seg()
    inner = Siamese()
    trainer = Trainer(bert_params, model_config, run_config, inner)

    def dataset_factory(input_files, is_for_training):
        return get_two_seg_data(input_files, run_config, model_config, is_for_training)

    tf_run(run_config, trainer, dataset_factory)


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)


