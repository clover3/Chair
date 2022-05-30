import sys

from taskman_client.wrapper3 import report_run3
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.dataset_factories import get_two_seg_data
from trainer_v2.custom_loop.neural_network_def.assymetric import ModelConfig2SegProject, BERTAsymmetricPMC
from trainer_v2.custom_loop.per_task.inner_network import AsymmetricMeanPool
from trainer_v2.custom_loop.run_config2 import get_run_config2_nli, RunConfig2
from trainer_v2.custom_loop.train_loop import tf_run_for_bert
from trainer_v2.train_util.arg_flags import flags_parser


@report_run3
def main(args):
    c_log.info("Start {}".format(__file__))
    run_config: RunConfig2 = get_run_config2_nli(args)

    model_config = ModelConfig2SegProject()
    inner = AsymmetricMeanPool(BERTAsymmetricPMC)

    def dataset_factory(input_files, is_for_training):
        return get_two_seg_data(input_files, run_config, model_config, is_for_training)

    tf_run_for_bert(dataset_factory, model_config, run_config, inner)


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)
