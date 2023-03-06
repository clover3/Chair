import sys

from cpath import get_bert_config_path
from taskman_client.wrapper3 import report_run3
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.dataset_factories import get_classification_dataset
from trainer_v2.custom_loop.modeling_common.bert_common import load_bert_config, ModelConfig300_3
from trainer_v2.custom_loop.neural_network_def.classification_trainer import StandardBertCls
from trainer_v2.custom_loop.run_config2 import get_run_config2_nli, RunConfig2, get_run_config2
from trainer_v2.custom_loop.train_loop import tf_run
from trainer_v2.custom_loop.trainer_if import TrainerIF
from trainer_v2.per_project.transparency.splade_regression.modeling.regression_modeling import get_regression_model
from trainer_v2.per_project.transparency.splade_regression.trainer_huggingface_init import TrainerHuggingfaceInit
from trainer_v2.train_util.arg_flags import flags_parser


@report_run3
def main(args):
    c_log.info(__file__)
    run_config: RunConfig2 = get_run_config2(args)
    run_config.print_info()

    model_config = None
    def model_factory():
        new_model = get_regression_model(run_config)
        return new_model

    trainer: TrainerIF = TrainerHuggingfaceInit(
        model_config, run_config, model_factory)

    def build_dataset(input_files, is_for_training):
        return NotImplemented

    tf_run(run_config, trainer, build_dataset)


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)


