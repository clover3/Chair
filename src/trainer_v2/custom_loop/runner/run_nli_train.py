import sys

from cpath import get_bert_config_path
from taskman_client.wrapper3 import report_run3
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.dataset_factories import get_classification_dataset
from trainer_v2.custom_loop.modeling_common.bert_common import load_bert_config, ModelConfig
from trainer_v2.custom_loop.per_task.classification_trainer import get_classification_trainer
from trainer_v2.custom_loop.run_config2 import get_run_config2_nli, RunConfig2
from trainer_v2.custom_loop.train_loop import tf_run_train
from trainer_v2.custom_loop.trainer_if import TrainerIF
from trainer_v2.train_util.arg_flags import flags_parser


@report_run3
def main(args):
    c_log.info("Start train Classification")
    run_config: RunConfig2 = get_run_config2_nli(args)
    run_config.print_info()

    bert_params = load_bert_config(get_bert_config_path())
    model_config = ModelConfig()
    trainer: TrainerIF = get_classification_trainer(bert_params, model_config, run_config)

    def build_dataset(input_files):
        return get_classification_dataset(input_files, run_config, model_config)

    tf_run_train(run_config, trainer, build_dataset)


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)


