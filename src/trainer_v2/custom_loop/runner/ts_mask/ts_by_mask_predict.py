import pickle
import sys

from taskman_client.wrapper3 import report_run3
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.dataset_factories import get_classification_dataset
from trainer_v2.custom_loop.definitions import ModelConfigType
from trainer_v2.custom_loop.inference import tf_run_predict
from trainer_v2.custom_loop.run_config2 import get_run_config2_nli, RunConfig2
from trainer_v2.train_util.arg_flags import flags_parser


class ModelConfig(ModelConfigType):
    max_seq_length = 300
    num_classes = 3
    num_local_classes = 3


@report_run3
def main(args):
    c_log.info("Start {}".format(__file__))
    run_config: RunConfig2 = get_run_config2_nli(args)
    run_config.print_info()
    model_config = ModelConfig()

    def build_dataset(input_files, is_for_training):
        return get_classification_dataset(input_files, run_config, model_config, is_for_training)

    prediction = tf_run_predict(run_config, build_dataset)
    pickle.dump(prediction, open(args.predict_save_path, "wb"))


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)


