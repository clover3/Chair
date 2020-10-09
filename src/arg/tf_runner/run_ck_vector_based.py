import tensorflow as tf

from taskman_client.wrapper import report_run
from tf_util.tf_logging import tf_logging, MuteEnqueueFilter
from tlm.model.vector_combining_model import vector_combining_model
from tlm.model_cnfig import JsonConfig
from tlm.training.input_fn import input_fn_builder_vector_ck
from tlm.training.train_config import TrainConfigEx
from tlm.training.train_flags import FLAGS
from trainer.tpu_estimator import run_estimator


@report_run
def main(_):
    tf_logging.info("Multi-evidence for QK")
    config = JsonConfig.from_json_file(FLAGS.model_config_file)
    train_config = TrainConfigEx.from_flags(FLAGS)

    input_fn = input_fn_builder_vector_ck(FLAGS, config)
    special_flags = []

    model_fn = vector_combining_model(config, train_config, special_flags)
    if FLAGS.do_predict:
        tf_logging.addFilter(MuteEnqueueFilter())

    return run_estimator(model_fn, input_fn)


if __name__ == "__main__":
    tf.compat.v1.app.run()

