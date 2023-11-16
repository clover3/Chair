import sys

from taskman_client.wrapper3 import report_run3
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.eval_loop import tf_run_eval
from trainer_v2.custom_loop.per_task.pairwise_trainer import TrainerForLossReturningModel
from trainer_v2.custom_loop.run_config2 import RunConfig2, get_run_config2
from trainer_v2.custom_loop.train_loop import tf_run_train
from trainer_v2.custom_loop.trainer_if import TrainerIFBase
from trainer_v2.per_project.transparency.mmp.pep.evidence_encoder.ee_train_common import EEEvalObject, \
    get_qd_multi_seg_dataset_from_model_config
from trainer_v2.per_project.transparency.mmp.pep.evidence_encoder.ee_train_model import EvidenceEncoder, \
    EEModelConfig32_2_20, loss_abs, EvidenceEncoderBaseline
from trainer_v2.train_util.arg_flags import flags_parser


@report_run3
def main(args):
    c_log.info(__file__)
    run_config: RunConfig2 = get_run_config2(args)
    run_config.print_info()

    model_config = EEModelConfig32_2_20()
    task_model = EvidenceEncoderBaseline(model_config, loss_abs)

    def build_dataset(input_files, is_for_training):
        dataset = get_qd_multi_seg_dataset_from_model_config(
            input_files, is_for_training, model_config, run_config)
        return dataset

    def eval_object_factory(model, eval_batches, dist_strategy, steps):
        return EEEvalObject(model_config, model, eval_batches, dist_strategy, steps)

    if run_config.is_training():
        trainer: TrainerIFBase = TrainerForLossReturningModel(run_config, task_model, eval_object_factory)
        return tf_run_train(run_config, trainer, build_dataset)
    else:
        evaler = NotImplemented
        metrics = tf_run_eval(run_config, evaler, build_dataset)
        return metrics


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)
