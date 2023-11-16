import tensorflow as tf
import sys

from taskman_client.wrapper3 import report_run3
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.dataset_factories import get_pairwise_dataset, get_qd_multi_seg_dataset
from trainer_v2.custom_loop.definitions import ModelConfig512_1
from trainer_v2.custom_loop.eval_loop import tf_run_eval
from trainer_v2.custom_loop.neural_network_def.two_seg_alt import CombineByScoreAdd
from trainer_v2.custom_loop.neural_network_def.two_seg_two_model import TwoSegConcatLogitCombineTwoModel
from trainer_v2.custom_loop.per_task.pairwise_trainer import TrainerForLossReturningModel, PairwiseEvaler
from trainer_v2.custom_loop.run_config2 import get_run_config2_train, RunConfig2, get_run_config2
from trainer_v2.custom_loop.train_loop import tf_run_train
from trainer_v2.custom_loop.trainer_if import TrainerIFBase
from trainer_v2.per_project.transparency.mmp.pep.evidence_encoder.ee_train_model import EvidenceEncoder, \
    EEModelConfig32_2_20, loss_per_qd
from trainer_v2.train_util.arg_flags import flags_parser


def convert_to_binary(item):
    scores = item['scores']
    threshold = 1.1
    binary_boolean = tf.less(threshold, scores)

    binary_i = tf.cast(binary_boolean, tf.int32)
    item["scores"] = binary_i
    return item


# NOT Implemented



@report_run3
def main(args):
    c_log.info(__file__)
    run_config: RunConfig2 = get_run_config2(args)
    run_config.print_info()

    model_config = EEModelConfig32_2_20()
    task_model = EvidenceEncoder(model_config, loss_per_qd, label_type='int32')

    def build_dataset(input_files, is_for_training):
        q_seq_len = model_config.max_num_qt * model_config.segment_len
        d_seq_len = model_config.max_num_dt * model_config.segment_len
        n_scores = model_config.max_num_qt * model_config.max_num_dt
        dataset = get_qd_multi_seg_dataset(
            input_files, run_config,
            q_seq_len, d_seq_len, n_scores,
            is_for_training)
        return dataset.map(convert_to_binary)

    if run_config.is_training():
        trainer: TrainerIFBase = TrainerForLossReturningModel(run_config, task_model)
        return tf_run_train(run_config, trainer, build_dataset)
    else:
        evaler = NotImplemented
        metrics = tf_run_eval(run_config, evaler, build_dataset)
        return metrics


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)
