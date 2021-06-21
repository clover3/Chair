
from my_tf import tf
from taskman_client.wrapper import report_run
from tlm.training.bert_with_new_ce import BertNewCE
from tlm.training.pointwise_train import run_pointwise_train
from tlm.training.train_flags import *


@report_run
def main(_):
    model_class = BertNewCE
    return run_pointwise_train(model_class)


if __name__ == "__main__":
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("model_config_file")
    flags.mark_flag_as_required("output_dir")
    flags.mark_flag_as_required("run_name")
    tf.compat.v1.app.run()

