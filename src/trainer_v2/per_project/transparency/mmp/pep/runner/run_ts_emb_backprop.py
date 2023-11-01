import sys

from cpath import get_bert_config_path
from taskman_client.wrapper3 import report_run3
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.dataset_factories import get_classification_dataset_hf_to_bert_f2
from trainer_v2.custom_loop.definitions import ModelConfig512_2
from trainer_v2.custom_loop.modeling_common.bert_common import load_bert_config
from trainer_v2.custom_loop.neural_network_def.combine_mat import MatrixCombineTwoWay
from trainer_v2.custom_loop.neural_network_def.ts_emb_backprop import TSEmbBackprop
from trainer_v2.custom_loop.per_task.pairwise_trainer import TrainerForLossReturningModel
from trainer_v2.custom_loop.prediction_trainer import ModelV2IF
from trainer_v2.custom_loop.run_config2 import get_run_config2_train, RunConfig2
from trainer_v2.custom_loop.train_loop import tf_run, tf_run_train
from trainer_v2.custom_loop.trainer_if import TrainerIFBase
from trainer_v2.train_util.arg_flags import flags_parser


@report_run3
def main(args):
    c_log.info(__file__)
    run_config: RunConfig2 = get_run_config2_train(args)
    run_config.print_info()

    bert_params = load_bert_config(get_bert_config_path())
    model_config = ModelConfig512_2()
    task_model: ModelV2IF = TSEmbBackprop(model_config, MatrixCombineTwoWay)
    trainer: TrainerIFBase = TrainerForLossReturningModel(run_config, task_model)

    def build_dataset(input_files, is_for_training):
        return NotImplemented

    return tf_run_train(run_config, trainer, build_dataset)


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)
