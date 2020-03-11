from tlm.model_cnfig import JsonConfig
from tlm.training.classification_model_fn import model_fn_classification
from tlm.training.flags_wrapper import input_fn_from_flags
from tlm.training.input_fn import input_fn_builder_classification
from tlm.training.train_config import TrainConfigEx
from tlm.training.train_flags import FLAGS
from trainer.tpu_estimator import run_estimator


def run_classification_task(model_class):
    config = JsonConfig.from_json_file(FLAGS.model_config_file)
    train_config = TrainConfigEx.from_flags(FLAGS)
    input_fn = input_fn_from_flags(input_fn_builder_classification, FLAGS)
    model_fn = model_fn_classification(config, train_config, model_class, FLAGS.special_flags.split(","))
    return run_estimator(model_fn, input_fn)