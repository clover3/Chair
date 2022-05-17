import logging
import sys

from absl import logging as absl_logging

from trainer_v2.chair_logging import c_log

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
c_log.info("import tensorflow ...")
import tensorflow as tf
c_log.info("import tensorflow DONE")
from trainer_v2.train_util.arg_flags import flags_parser
from trainer_v2.partial_processing.model_fn_builder import model_factory_cls
from trainer_v2.partial_processing.config_helper import get_model_config_nli
from trainer_v2.partial_processing.input_fn import create_classifier_dataset
from trainer_v2.train_util.input_fn_common import get_input_fn
from trainer_v2.partial_processing.run_bert_based_classifier import run_classification
from trainer_v2.run_config import RunConfigEx, get_run_config_nli_train

from tensorflow.python.data import Dataset

def main(args):
    dummy_loggin = absl_logging
    c_log.setLevel(logging.DEBUG)
    c_log.info("dev_run_train main.py")
    model_config = get_model_config_nli()
    run_config: RunConfigEx = get_run_config_nli_train(args)
    get_model_fn = model_factory_cls(model_config, run_config)

    def get_dataset(input_files, is_training) -> Dataset:
        dataset = create_classifier_dataset(
            tf.io.gfile.glob(input_files),
            model_config.max_seq_length,
            run_config.batch_size,
            is_training=is_training)
        return dataset

    train_input_fn, eval_input_fn = get_input_fn(args, get_dataset)
    run_classification(args, run_config, get_model_fn, train_input_fn, eval_input_fn)


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)
