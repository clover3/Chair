import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from trainer_v2.custom_loop.neural_network_def.combine2d import ReduceMaxLayer
from trainer_v2.custom_loop.neural_network_def.inferred_attention import InferredAttention
from trainer_v2.custom_loop.neural_network_def.segmented_enc import FuzzyLogicLayerNoSum
from cpath import get_bert_config_path
from trainer_v2.custom_loop.modeling_common.bert_common import load_bert_config
from trainer_v2.custom_loop.per_task.reference_model import RefModelTrainer
from trainer_v2.custom_loop.train_loop import tf_run
from trainer_v2.custom_loop.train_loop_addon import adjust_logging
from taskman_client.wrapper3 import report_run3
from trainer_v2.custom_loop.dataset_factories import get_two_seg_data
from trainer_v2.custom_loop.neural_network_def.siamese import ModelConfig2SegProject
from trainer_v2.custom_loop.run_config2 import RunConfig2, get_run_config2_nli
from trainer_v2.train_util.arg_flags import flags_parser
from trainer_v2.chair_logging import c_log


@report_run3
def main(args):
    run_config: RunConfig2 = get_run_config2_nli(args)
    model_config = ModelConfig2SegProject()

    def dataset_factory(input_files, is_for_training):
        return get_two_seg_data(input_files, run_config, model_config, is_for_training)

    adjust_logging()
    run_config.print_info()
    model = InferredAttention(ReduceMaxLayer, FuzzyLogicLayerNoSum)
    bert_params = load_bert_config(get_bert_config_path())
    trainer = RefModelTrainer(bert_params, model_config, run_config, model)
    tf_run(run_config, trainer, dataset_factory)


if __name__ == "__main__":
    c_log.info("Start {}".format(__file__))
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)
