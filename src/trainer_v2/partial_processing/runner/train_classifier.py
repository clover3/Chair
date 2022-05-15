import sys

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from trainer_v2.arg_flags import flags_parser
from trainer_v2.chair_logging import c_log

c_log.info("Loading tensorflow ...")
c_log.info("Loading tensorflow DONE")
from trainer_v2.partial_processing.assymetric_model import model_factory_cls
from trainer_v2.partial_processing.config_helper import get_model_config_nli
from trainer_v2.partial_processing.input_fn import build_classification_dataset
from trainer_v2.partial_processing.misc_helper import parse_input_files
from trainer_v2.partial_processing.run_bert_based_classifier import run_classification
from trainer_v2.run_config import RunConfigEx, get_run_config_nli_train


def get_processor(args):
    if args.use_tpu:
        return "tpu"
    else:
        return "gpu"


def main(args):
    c_log.info("dev_run_train main.py")
    model_config = get_model_config_nli()
    run_config: RunConfigEx = get_run_config_nli_train(args)
    get_model_fn = model_factory_cls(model_config, run_config)
    is_training = True

    def train_input_fn():
        input_files = parse_input_files(args.input_files)
        dataset = build_classification_dataset(model_config, input_files, run_config, is_training)
        return dataset

    def eval_input_fn():
        if args.eval_input_files is None:
            return None
        input_files = parse_input_files(args.eval_input_files)
        dataset = build_classification_dataset(model_config, input_files, run_config, False)
        return dataset

    run_classification(args, run_config, get_model_fn, train_input_fn, eval_input_fn)


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)
