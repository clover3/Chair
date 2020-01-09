import tensorflow as tf

import tlm.model.base as modeling
from tlm.training.train_flags import *
from trainer.estimator_main_v2 import main_inner


def main(_):
    return main_inner(modeling.BertModelMoreFF)

if __name__ == "__main__":
    flags.mark_flag_as_required("input_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.compat.v1.app.run()
