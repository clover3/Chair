import logging
import sys

from omegaconf import OmegaConf

from taskman_client.wrapper3 import JobContext
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.definitions import ModelConfigType
from trainer_v2.custom_loop.eval_loop import tf_run_eval
from trainer_v2.custom_loop.per_task.pairwise_trainer import TrainerForLossReturningModel, LossOnlyEvalObject
from trainer_v2.custom_loop.run_config2 import RunConfig2, CommonRunConfig, TrainConfig, DeviceConfig, DatasetConfig
from trainer_v2.custom_loop.train_loop import tf_run_train
from trainer_v2.per_project.transparency.mmp.pep_to_tt.dataset_builder import PEP_TT_DatasetBuilder, \
    PEP_TT_DatasetBuilderWithAlignInfo, get_pep_tt_single_encoder_for_with_align_info
from trainer_v2.per_project.transparency.mmp.pep_to_tt.pep_tt_modeling import PEP_TT_ModelConfig, \
    PEP_TT_Model_Single, PEP_TT_Model_Single2
from trainer_v2.per_project.transparency.mmp.pep_to_tt.runner2.train_with_align_info import get_run_config_no_inputs


class PEP_TT_ModelConfig2(PEP_TT_ModelConfig):
    max_seq_length = 16
    num_classes = 1
    max_num_terms = 10
    reg_weight = 0.1


def main():
    c_log.info(__file__)
    conf = OmegaConf.load(sys.argv[1])
    run_config: RunConfig2 = get_run_config_no_inputs(conf)

    with JobContext(run_config.common_run_config.run_name):
        model_config = PEP_TT_ModelConfig2()
        task_model = PEP_TT_Model_Single2(model_config)
        encoder = get_pep_tt_single_encoder_for_with_align_info(model_config, conf)
        builder = PEP_TT_DatasetBuilderWithAlignInfo(
            encoder,
            conf,
            run_config.common_run_config.batch_size)

        if run_config.is_training():
            trainer: TrainerForLossReturningModel = \
                TrainerForLossReturningModel(run_config, task_model, LossOnlyEvalObject)
            ret = tf_run_train(run_config, trainer, builder.get_pep_tt_dataset)
        else:
            evaler = NotImplemented
            metrics = tf_run_eval(run_config, evaler, builder.get_pep_tt_dataset)
            return metrics


if __name__ == "__main__":
    main()
