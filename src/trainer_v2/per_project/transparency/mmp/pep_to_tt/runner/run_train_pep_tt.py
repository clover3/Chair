import sys

from omegaconf import OmegaConf

from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.eval_loop import tf_run_eval
from trainer_v2.custom_loop.per_task.pairwise_trainer import TrainerForLossReturningModel, PairwiseEvaler
from trainer_v2.custom_loop.run_config2 import RunConfig2
from trainer_v2.custom_loop.train_loop import tf_run_train
from trainer_v2.per_project.transparency.mmp.pep_to_tt.dataset_builder import PEP_TT_DatasetBuilder
from trainer_v2.per_project.transparency.mmp.pep_to_tt.pep_tt_encoders import PEP_TT_EncoderMulti
from trainer_v2.per_project.transparency.mmp.pep_to_tt.omega_conf_run_config import get_run_config_from_omega_conf
from trainer_v2.per_project.transparency.mmp.pep_to_tt.pep_tt_modeling import PEP_TT_ModelConfig, \
    PEP_TT_Model


def main():
    c_log.info(__file__)
    conf = OmegaConf.load(sys.argv[1])
    run_config: RunConfig2 = get_run_config_from_omega_conf(conf)
    model_config = PEP_TT_ModelConfig()

    task_model = PEP_TT_Model(model_config)
    encoder = PEP_TT_EncoderMulti(model_config, conf)
    builder = PEP_TT_DatasetBuilder(encoder, run_config.common_run_config.batch_size)

    trainer: TrainerForLossReturningModel = TrainerForLossReturningModel(run_config, task_model)
    return tf_run_train(run_config, trainer, builder.get_pep_tt_dataset)

if __name__ == "__main__":
    main()
