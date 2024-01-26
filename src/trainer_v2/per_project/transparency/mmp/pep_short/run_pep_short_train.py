# Input Format : (qt, (dt1_highest, s1), (dt2, s2), (dt3, s3))
# Loss function,
import logging

from data_generator.tokenizer_wo_tf import get_tokenizer
from trainer_v2.per_project.transparency.mmp.pep_short.dataset_builder import PepShortEncoder, PEPShortDatasetBuilder

import sys

from omegaconf import OmegaConf

from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.eval_loop import tf_run_eval
from trainer_v2.custom_loop.per_task.pairwise_trainer import TrainerForLossReturningModel, PairwiseEvaler, \
    LossOnlyEvalObject
from trainer_v2.custom_loop.train_loop import tf_run_train
from trainer_v2.custom_loop.run_config2 import RunConfig2, CommonRunConfig, TrainConfig, DeviceConfig, \
    DatasetConfig
from trainer_v2.per_project.transparency.mmp.pep_short.pep_short_modeling import PEPShortModelConfig, PEP_TTShort


def get_run_config(omega_conf):
    common_run_config = CommonRunConfig(batch_size=omega_conf.batch_size)
    train_config = TrainConfig(
        train_step=omega_conf.train_step,
        save_every_n_step=omega_conf.save_every_n_step,
        eval_every_n_step=omega_conf.eval_every_n_step,
        model_save_path=omega_conf.model_save_path,
        init_checkpoint=omega_conf.init_checkpoint
    )
    device_config = DeviceConfig()
    dataset_config = DatasetConfig(
        omega_conf.dataset_path,
        omega_conf.eval_dataset_path
    )
    run_config = RunConfig2(common_run_config=common_run_config,
                            dataset_config=dataset_config,
                            train_config=train_config,
                            device_config=device_config
                            )
    run_config.print_info()
    return run_config


def main():
    c_log.info(__file__)
    c_log.setLevel(logging.INFO)
    conf = OmegaConf.load(sys.argv[1])
    run_config: RunConfig2 = get_run_config(conf)
    model_config = PEPShortModelConfig()
    task_model = PEP_TTShort(model_config)

    encoder = PepShortEncoder(get_tokenizer(), model_config)
    builder = PEPShortDatasetBuilder(encoder, run_config.common_run_config.batch_size)

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
