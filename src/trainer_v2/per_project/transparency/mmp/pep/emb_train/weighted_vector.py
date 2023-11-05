import sys

from taskman_client.wrapper3 import report_run3
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.definitions import ModelConfig512_1
from trainer_v2.custom_loop.neural_network_def.ts_emb_backprop import TSEmbBackprop, TSEmbWeights
from trainer_v2.custom_loop.neural_network_def.two_seg_alt import CombineByScoreAdd
from trainer_v2.custom_loop.run_config2 import get_run_config2_train, RunConfig2
from trainer_v2.custom_loop.train_loop import tf_run_train
from trainer_v2.per_project.transparency.mmp.pep.emb_train.embedding_trainer import EmbeddingTrainer
from trainer_v2.train_util.arg_flags import flags_parser


@report_run3
def main(args):
    c_log.info(__file__)
    run_config: RunConfig2 = get_run_config2_train(args)
    run_config.print_info()

    model_config = ModelConfig512_1()
    task_model: TSEmbWeights = TSEmbWeights(model_config, CombineByScoreAdd)
    target_q_token = "when"
    trainer: EmbeddingTrainer = EmbeddingTrainer(
        run_config, task_model, target_q_token, model_config)

    def build_dataset(input_files, is_for_training):
        # Train data indicate Template for (qt, dt)
        # Eg., (when, [SPE1])
        # E.g, (when, in [SEP1])
        return trainer.build_dataset_spe1(is_for_training)

    return tf_run_train(run_config, trainer, build_dataset)


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)
