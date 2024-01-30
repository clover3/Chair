import logging
import sys

from taskman_client.wrapper3 import report_run3
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.eval_loop import tf_run_eval
from trainer_v2.custom_loop.per_task.pairwise_trainer import TrainerForLossReturningModel, PairwiseEvaler, \
    LossOnlyEvalObject
from trainer_v2.custom_loop.run_config2 import RunConfig2, get_run_config2
from trainer_v2.custom_loop.train_loop import tf_run_train
from trainer_v2.per_project.transparency.mmp.pep_to_tt.dataset_builder import read_pep_tt_dataset
from trainer_v2.per_project.transparency.mmp.pep_to_tt.pep_tt_modeling import PEP_TT_ModelConfig, \
    PEP_TT_Model_Single2
from trainer_v2.train_util.arg_flags import flags_parser


@report_run3
def main(args):
    c_log.info(__file__)
    c_log.setLevel(logging.INFO)
    run_config: RunConfig2 = get_run_config2(args)
    run_config.print_info()
    model_config = PEP_TT_ModelConfig()
    c_log.info("Train with 0 reg loss")
    task_model = PEP_TT_Model_Single2(model_config)
    seq_len = model_config.max_seq_length

    def build_dataset(input_files, is_for_training):
        return read_pep_tt_dataset(
            input_files, run_config, seq_len, is_for_training)

    if run_config.is_training():
        trainer: TrainerForLossReturningModel = \
            TrainerForLossReturningModel(run_config, task_model, LossOnlyEvalObject)
        return tf_run_train(run_config, trainer, build_dataset)
    else:
        evaler = PairwiseEvaler(run_config)
        metrics = tf_run_eval(run_config, evaler, build_dataset)
        return metrics


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)
