from trainer_v2.custom_loop.run_config2 import RunConfig2, get_run_config2
from trainer_v2.custom_loop.train_loop import tf_run2
from trainer_v2.custom_loop.trainer_if import TrainerIFBase
from trainer_v2.per_project.transparency.mmp.alignment.dataset_factory import read_galign_pairwise
from trainer_v2.per_project.transparency.mmp.trainer_d_out2 import TrainerDOut2


def run_galign_pairwise_training(args, model_v3):
    run_config: RunConfig2 = get_run_config2(args)
    run_config.print_info()
    trainer: TrainerIFBase = TrainerDOut2(run_config, model_v3)

    def build_dataset(input_files, is_for_training):
        return read_galign_pairwise(
            input_files, run_config, is_for_training)

    tf_run2(run_config, trainer, build_dataset)