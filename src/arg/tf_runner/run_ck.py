from my_tf import tf
from taskman_client.wrapper import report_run
from tlm.training.run_classification_w_second_input import run_classification_w_second_input
from tlm.training.train_flags import *


@report_run
def main(_):
    return run_classification_w_second_input()


if __name__ == "__main__":
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    flags.mark_flag_as_required("run_name")
    tf.compat.v1.app.run()
