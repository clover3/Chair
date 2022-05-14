import sys

from trainer_v2.arg_flags import flags_parser
from trainer_v2.chair_logging import c_log
from trainer_v2.partial_processing.assymetric_model import model_fn_factory
from trainer_v2.partial_processing.config_helper import get_run_config_nli_train, MultiSegModelConfig
from trainer_v2.partial_processing.input_fn import create_two_seg_classification_dataset
from trainer_v2.partial_processing.misc_helper import parse_input_files
from trainer_v2.partial_processing.run_bert_based_classifier import run_keras_compile_fit_wrap, \
    get_bert_config
from trainer_v2.run_config import RunConfigEx


def get_model_config(max_seq_length_list):
    bert_config = get_bert_config()
    model_config = MultiSegModelConfig(bert_config, max_seq_length_list)
    return model_config


def main(args):
    c_log.info("two_seg_classifier main.py")
    max_seq_length_list = [200, 100]
    model_config = get_model_config(max_seq_length_list)
    run_config: RunConfigEx = get_run_config_nli_train(args)
    run_config.batch_size = 8
    run_config.steps_per_execution = 10
    get_model_fn = model_fn_factory(model_config, run_config)
    is_training = True

    def train_input_fn():
        input_files = parse_input_files(args.input_files)
        dataset = create_two_seg_classification_dataset(input_files,
                                                        max_seq_length_list,
                                                        run_config.batch_size,
                                                        is_training)
        return dataset

    run_keras_compile_fit_wrap(args, get_model_fn, run_config, train_input_fn)


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)
