import logging
import sys

from cpath import get_bert_config_path
from taskman_client.wrapper3 import report_run3
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.dataset_factories import get_classification_dataset
from trainer_v2.custom_loop.definitions import ModelConfigType
from trainer_v2.custom_loop.modeling_common.bert_common import load_bert_config
from trainer_v2.custom_loop.neural_network_def.combine_mat import MatrixCombineTrainable, MatrixCombine
from trainer_v2.custom_loop.neural_network_def.single_seg_w_role import SingleSegWRole
from trainer_v2.custom_loop.neural_network_def.var_local_decisions import SingleVarLD
from trainer_v2.custom_loop.per_task.trainer import Trainer
from trainer_v2.custom_loop.run_config2 import get_run_config2_nli, RunConfig2
from trainer_v2.custom_loop.train_loop import tf_run
from trainer_v2.custom_loop.trainer_if import TrainerIF
from trainer_v2.train_util.arg_flags import flags_parser


class ModelConfig(ModelConfigType):
    max_seq_length = 300
    num_classes = 3
    num_local_classes = 3


@report_run3
def main(args):
    c_log.info("Start {}".format(__file__))
    run_config: RunConfig2 = get_run_config2_nli(args)
    run_config.print_info()

    bert_params = load_bert_config(get_bert_config_path())
    model_config = ModelConfig()
    task_model = SingleSegWRole(MatrixCombine)
    trainer: TrainerIF = Trainer(bert_params, model_config, run_config, task_model)

    def build_dataset(input_files, is_for_training):
        return get_classification_dataset(input_files, run_config, model_config, is_for_training)

    msg = tf_run(run_config, trainer, build_dataset)


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)


