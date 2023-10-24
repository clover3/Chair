import sys

from taskman_client.wrapper3 import report_run3
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.dataset_factories import get_pairwise_dataset
from trainer_v2.custom_loop.definitions import ModelConfig512_1
from trainer_v2.custom_loop.eval_loop import tf_run_eval
from trainer_v2.custom_loop.neural_network_def.two_seg_alt import CombineByScoreAdd
from trainer_v2.custom_loop.neural_network_def.two_seg_two_model import TwoSegConcatLogitCombineTwoModel
from trainer_v2.custom_loop.per_task.pairwise_trainer import PairwiseTrainer, PairwiseEvaler
from trainer_v2.custom_loop.run_config2 import get_run_config2_train, RunConfig2, get_run_config2
from trainer_v2.custom_loop.train_loop import tf_run_train
from trainer_v2.custom_loop.trainer_if import TrainerIFBase
from trainer_v2.train_util.arg_flags import flags_parser


@report_run3
def main(args):
    c_log.info(__file__)
    run_config: RunConfig2 = get_run_config2(args)
    run_config.print_info()

    model_config = ModelConfig512_1()
    task_model = TwoSegConcatLogitCombineTwoModel(model_config, CombineByScoreAdd)

    def build_dataset(input_files, is_for_training):
        return get_pairwise_dataset(
            input_files, run_config, model_config, is_for_training,
            add_dummy_y=False,
            segment_ids_for_token_type_ids=True)

    if run_config.is_training():
        trainer: TrainerIFBase = PairwiseTrainer(run_config, task_model)
        return tf_run_train(run_config, trainer, build_dataset)
    else:
        evaler = PairwiseEvaler(run_config)
        return tf_run_eval(run_config, evaler, build_dataset)


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)


