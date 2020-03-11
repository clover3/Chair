import tensorflow as tf

from taskman_client.wrapper import report_run
from tf_util.tf_logging import tf_logging
from tlm.model.dual_bert import DualBertModel
from tlm.training.classification_common import run_classification_task
from tlm.training.train_flags import *


@report_run
def main(_):
    tf_logging.info("run_dual_bert")
    return run_classification_task(DualBertModel)


if __name__ == "__main__":
    flags.mark_flag_as_required("model_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.compat.v1.app.run()

