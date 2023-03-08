import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from trainer_v2.custom_loop.neural_network_def.inner_network import ClassificationModelIF
from trainer_v2.custom_loop.neural_network_def.siamese import ModelConfig200_200
from taskman_client.wrapper3 import report_run3
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.dataset_factories import get_two_seg_data
from trainer_v2.custom_loop.neural_network_def.segmented_enc import MLPLabelCombine
from trainer_v2.custom_loop.neural_network_def.asymmetric2 import BERTEvenSegmented
from trainer_v2.custom_loop.run_config2 import get_run_config2_nli, RunConfig2
from trainer_v2.custom_loop.train_loop_addon import tf_run_for_bert
from trainer_v2.train_util.arg_flags import flags_parser


@report_run3
def main(args):
    c_log.info("Start {}".format(__file__))
    run_config: RunConfig2 = get_run_config2_nli(args)

    model_config = ModelConfig200_200()

    def combine_fn():
        return MLPLabelCombine(2, model_config.num_classes)
    inner: ClassificationModelIF = BERTEvenSegmented(combine_fn)

    def dataset_factory(input_files, is_for_training):
        return get_two_seg_data(input_files, run_config, model_config, is_for_training)

    tf_run_for_bert(dataset_factory, model_config, run_config, inner)


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)

