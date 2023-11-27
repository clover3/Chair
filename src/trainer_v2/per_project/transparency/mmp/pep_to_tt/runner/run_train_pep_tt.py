import sys

from omegaconf import OmegaConf

from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.eval_loop import tf_run_eval
from trainer_v2.custom_loop.per_task.pairwise_trainer import TrainerForLossReturningModel
from trainer_v2.custom_loop.run_config2 import RunConfig2, CommonRunConfig, TrainConfig, DeviceConfig, \
    DatasetConfig
from trainer_v2.custom_loop.train_loop import tf_run_train
from trainer_v2.per_project.transparency.mmp.pep_to_tt.dataset_builder import PEP_TT_DatasetBuilder, \
    PEP_TT_EncoderMulti
from trainer_v2.per_project.transparency.mmp.pep_to_tt.pep_tt_modeling import PEP_TT_ModelConfig, \
    PEP_TT_Model


def get_run_config(omega_conf):
    common_run_config = CommonRunConfig(batch_size=omega_conf.batch_size)

    train_config = TrainConfig(
        train_step=omega_conf.train_step,
        save_every_n_step=omega_conf.save_every_n_step,
        eval_every_n_step=omega_conf.eval_every_n_step,
        init_checkpoint=omega_conf.init_checkpoint
    )
    device_config = DeviceConfig()
    dataset_config = DatasetConfig(omega_conf.dataset_path, omega_conf.dataset_path)
    run_config = RunConfig2(common_run_config=common_run_config,
                            dataset_config=dataset_config,
                            train_config=train_config,
                            device_config=device_config
                            )

    return run_config


def main():
    c_log.info(__file__)
    conf = OmegaConf.load(sys.argv[1])
    run_config: RunConfig2 = get_run_config(conf)
    model_config = PEP_TT_ModelConfig()

    task_model = PEP_TT_Model(model_config)
    encoder = PEP_TT_EncoderMulti(model_config, conf)
    builder = PEP_TT_DatasetBuilder(encoder, run_config.common_run_config.batch_size)

    if run_config.is_training():
        trainer: TrainerForLossReturningModel = TrainerForLossReturningModel(run_config, task_model)
        return tf_run_train(run_config, trainer, builder.get_pep_tt_dataset)
    else:
        evaler = NotImplemented
        metrics = tf_run_eval(run_config, evaler, builder.get_pep_tt_dataset)
        return metrics


if __name__ == "__main__":
    main()
