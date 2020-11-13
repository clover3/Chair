import tensorflow as tf

from taskman_client.wrapper import report_run
from tf_util.tf_logging import tf_logging, MuteEnqueueFilter
from tlm.model.base import BertModel
from tlm.model.dot_product_model import model_fn_pointwise_ranking
from tlm.model.projected_max_pooling import ProjectedMaxPooling
from tlm.model_cnfig import JsonConfig
from tlm.training.input_fn import input_fn_builder_dot_product_ck
from tlm.training.train_config import TrainConfigEx
from tlm.training.train_flags import *
from tlm.training.train_flags import FLAGS
from trainer.tpu_estimator import run_estimator


@report_run
def main(_):
    tf_logging.info("Multi-evidence with Dot Product (CK version)")
    config = JsonConfig.from_json_file(FLAGS.model_config_file)
    train_config = TrainConfigEx.from_flags(FLAGS)
    total_doc_length = config.max_doc_length * config.num_docs
    input_fn = input_fn_builder_dot_product_ck(FLAGS, config.max_sent_length, total_doc_length)
    special_flags = FLAGS.special_flags.split(",")
    special_flags.append("feed_features")
    modeling = FLAGS.modeling
    if modeling is None:
        model_class = BertModel
    elif modeling == "pmp":
        model_class = ProjectedMaxPooling
    else:
        assert False

    model_fn = model_fn_pointwise_ranking(config, train_config, model_class, special_flags)
    if FLAGS.do_predict:
        tf_logging.addFilter(MuteEnqueueFilter())

    return run_estimator(model_fn, input_fn)


if __name__ == "__main__":
    flags.mark_flag_as_required("model_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.compat.v1.app.run()

