import sys

from trainer_v2.arg_flags import flags_parser
from trainer_v2.chair_logging import c_log
from trainer_v2.partial_processing.assymetric_model import model_fn_factory
from trainer_v2.partial_processing.config_helper import get_run_config_nli_train
from trainer_v2.partial_processing.input_fn import build_classification_dataset
from trainer_v2.partial_processing.misc_helper import parse_input_files
from trainer_v2.partial_processing.run_bert_based_classifier import get_model_config, run_keras_compile_fit_wrap
from trainer_v2.run_config import RunConfigEx


def main(args):
    c_log.info("dev_run_train main.py")
    model_config = get_model_config()
    run_config: RunConfigEx = get_run_config_nli_train(args)
    run_config.steps_per_execution = 1
    get_model_fn = model_fn_factory(model_config, run_config)
    is_training = True

    def train_input_fn():
        input_files = parse_input_files(args.input_files)
        dataset = build_classification_dataset(model_config, input_files, run_config, is_training)
        return dataset

    run_keras_compile_fit_wrap(args, get_model_fn, run_config, train_input_fn)


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)
