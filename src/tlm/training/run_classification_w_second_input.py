from tf_util.tf_logging import tf_logging, MuteEnqueueFilter
from tlm.model.base import BertConfig, BertModel
from tlm.training.classification_model_fn import model_fn_classification
from tlm.training.flags_wrapper import get_input_files_from_flags, show_input_files
from tlm.training.input_fn import input_fn_builder_use_second_input
from tlm.training.train_config import TrainConfigEx
from tlm.training.train_flags import *
from trainer.tpu_estimator import run_estimator


def run_classification_w_second_input():
    input_files = get_input_files_from_flags(FLAGS)
    bert_config = BertConfig.from_json_file(FLAGS.bert_config_file)
    train_config = TrainConfigEx.from_flags(FLAGS)
    show_input_files(input_files)
    special_flags = FLAGS.special_flags.split(",")
    model_fn = model_fn_classification(
        bert_config=bert_config,
        train_config=train_config,
        model_class=BertModel,
        special_flags=special_flags,
    )
    if FLAGS.do_predict:
        tf_logging.addFilter(MuteEnqueueFilter())
    input_fn = input_fn_builder_use_second_input(FLAGS)
    result = run_estimator(model_fn, input_fn)
    return result

