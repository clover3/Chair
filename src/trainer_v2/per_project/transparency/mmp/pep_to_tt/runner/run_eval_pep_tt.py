import sys

from omegaconf import OmegaConf

from misc_lib import path_join
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.eval_loop import tf_run_eval
from trainer_v2.custom_loop.per_task.pairwise_trainer import PairwiseEvaler
from trainer_v2.custom_loop.run_config2 import RunConfig2
from trainer_v2.per_project.transparency.mmp.pep_to_tt.dataset_builder import PEP_TT_DatasetBuilder, \
    PEP_TT_EncoderMulti
from trainer_v2.per_project.transparency.mmp.pep_to_tt.omega_conf_run_config import get_eval_run_config_from_omega_conf
from trainer_v2.per_project.transparency.mmp.pep_to_tt.pep_tt_modeling import PEP_TT_ModelConfig


def main():
    c_log.info(__file__)
    conf = OmegaConf.load(sys.argv[1])
    step = int(sys.argv[2])
    conf.model_save_path = path_join(conf.model_save_dir, f"model_{step}")
    conf.run_name = conf.run_name + f"_{step}"
    run_config: RunConfig2 = get_eval_run_config_from_omega_conf(conf)
    run_config.common_run_config.report_field = "loss"


    model_config = PEP_TT_ModelConfig()
    encoder = PEP_TT_EncoderMulti(model_config, conf)
    builder = PEP_TT_DatasetBuilder(encoder, run_config.common_run_config.batch_size)
    evaler = PairwiseEvaler(run_config)
    metrics = tf_run_eval(run_config, evaler, builder.get_pep_tt_dataset)
    return metrics


if __name__ == "__main__":
    main()
