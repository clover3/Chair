import sys

from trainer_v2.arg_flags import flags_parser
from trainer_v2.chair_logging import c_log
from trainer_v2.partial_processing.assymetric_model import model_factory_assym
from trainer_v2.partial_processing.config_helper import MultiSegModelConfig, get_bert_config
from trainer_v2.partial_processing.input_fn import create_two_seg_classification_dataset
from trainer_v2.partial_processing.misc_helper import parse_input_files
from trainer_v2.partial_processing.run_bert_based_classifier import run_classification
from trainer_v2.run_config import RunConfigEx, get_run_config_nli_train


def get_model_config():
    bert_config = get_bert_config()
    max_seq_length_list = [200, 100]
    model_config = MultiSegModelConfig(bert_config, max_seq_length_list)
    return model_config


def main(args):
    c_log.info("two_seg_classifier main.py")
    model_config = get_model_config()
    run_config: RunConfigEx = get_run_config_nli_train(args)
    get_model_fn = model_factory_assym(model_config, run_config)
    is_training = True

    def train_input_fn():
        input_files = parse_input_files(args.input_files)
        dataset = create_two_seg_classification_dataset(input_files,
                                                        model_config.max_seq_length_list,
                                                        run_config.batch_size,
                                                        is_training)
        return dataset

    run_classification(args, run_config, get_model_fn, train_input_fn)


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)
