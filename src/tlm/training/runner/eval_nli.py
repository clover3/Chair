import tensorflow as tf

from taskman_client.wrapper import report_run
from tlm.training.train_flags import *
from trainer.estimator_main_v2 import main_inner


@report_run
def main(_):
    return main_inner()

if __name__ == "__main__":
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    flags.mark_flag_as_required("run_name")
    tf.compat.v1.app.run()
