import os
import sys
from collections import Counter

from trainer_v2.custom_loop.modeling_common.tf_helper import distribute_dataset
from trainer_v2.custom_loop.train_loop_helper import get_strategy_from_config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from taskman_client.wrapper3 import report_run3
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.dataset_factories import get_two_seg_data
from trainer_v2.custom_loop.neural_network_def.assymetric import ModelConfig2SegProject
from trainer_v2.custom_loop.run_config2 import get_run_config2_nli, RunConfig2
from trainer_v2.train_util.arg_flags import flags_parser


@report_run3
def main(args):
    c_log.info("Start {}".format(__file__))
    run_config: RunConfig2 = get_run_config2_nli(args)

    model_config = ModelConfig2SegProject()

    def dataset_factory(input_files, is_for_training):
        return get_two_seg_data(input_files, run_config, model_config, is_for_training)

    eval_dataset = dataset_factory(run_config.dataset_config.eval_files_path, False)
    strategy = get_strategy_from_config(run_config)
    eval_batches = distribute_dataset(strategy, eval_dataset)
    max_step = sum(1 for _ in eval_batches)
    iterator = iter(eval_batches)
    counter = Counter()
    for idx in range(max_step):
        args = next(iterator)
        x, y = args
        for label in y:
            label_i = int(label)
            counter[label_i] += 1

    print(counter)

    s = sum(counter.values())
    for label in counter:
        print(label, counter[label]/s)


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)
